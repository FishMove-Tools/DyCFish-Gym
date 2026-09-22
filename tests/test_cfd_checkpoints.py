"""CFD resume integrity and raw-return ranking without ROM dependencies."""
from pathlib import Path
import csv
import pytest
from CFD_stage import training as cfd
from CFD_stage import checkpoints as train


def test_resume_snapshot_survives_global_replacement_after_unlock(tmp_path, monkeypatch):
    global_dir = tmp_path / "saved_models"
    global_dir.mkdir()
    model, stats = global_dir / "saved_model.zip", global_dir / "saved_vecnormalize.pkl"
    model.write_bytes(b"first model")
    stats.write_bytes(b"first stats")
    expected = cfd.cfd_contract([1, 0], "setup:initialize")
    train.write_metadata(model, stats, expected, 0, 10)

    class Lock:
        held = False
        def __enter__(self):
            self.held = True
        def __exit__(self, *exc):
            self.held = False
            # Simulate a different worker publishing a new best immediately.
            model.write_bytes(b"second model")
            stats.write_bytes(b"second stats")
            train.write_metadata(model, stats, expected, 1, 20)

    class Manager:
        lock = Lock()

    manager = Manager()
    original_copy = cfd.shutil.copy2
    copied = []
    def guarded_copy(source, destination):
        assert manager.lock.held, "Every companion file must be copied under the shared writer lock"
        copied.append(Path(source).name)
        return original_copy(source, destination)
    monkeypatch.setattr(cfd.shutil, "copy2", guarded_copy)
    local_model, local_stats = cfd.snapshot_resume_bundle(global_dir / "worker_0", tmp_path, expected, manager)
    assert len(copied) == 3
    assert local_model.read_bytes() == b"first model"
    assert local_stats.read_bytes() == b"first stats"
    assert model.read_bytes() == b"second model"
    loaded = train.validate_bundle(local_model, local_stats, local_model.with_suffix(".metadata.json"), expected)
    assert loaded["normalization_file"] == local_stats.name
    assert loaded["model_file"] == local_model.name


def test_best_model_and_episode_logs_use_monitor_raw_reward(tmp_path):
    class Base:
        def __init__(self, verbose):
            self.locals = {}
    class Manager:
        best = -float("inf")
        winner = None
        def update_worker_status(self, *args):
            pass
        def update_best_model(self, rank, reward, path):
            if reward > self.best:
                self.best, self.winner = reward, rank
            return False  # No actual shared checkpoint is written in this test.
        def get_best_reward(self):
            return self.best
    class Callback(cfd.CallbackLogic, Base):
        def _save_checkpoint(self, *args):
            pass

    manager = Manager()
    for rank, normalized_reward, raw_reward in [(0, 100.0, -50.0), (1, -100.0, 25.0)]:
        callback = Callback(str(tmp_path / f"worker_{rank}"), rank, manager, {}, rank, {})
        callback.locals = {"rewards": [normalized_reward], "dones": [True],
                           "infos": [{"episode": {"r": raw_reward, "l": 1}}]}
        assert callback._on_step()
        assert callback.best_mean_reward == raw_reward
        assert callback.episode_rewards == [raw_reward]
        with Path(callback.reward_log_path).open(newline="") as stream:
            row = next(csv.DictReader(stream))
        assert float(row["Raw_Reward"]) == raw_reward
        assert float(row["Mean_Raw_Reward"]) == raw_reward
    # Normalized rewards rank worker 0 above 1; raw returns correctly reverse it.
    assert manager.winner == 1
    assert manager.best == 25.0


def test_missing_monitor_return_stops_instead_of_ranking_normalized_reward(tmp_path):
    class Base:
        def __init__(self, verbose):
            self.locals = {}
    class Manager:
        state = None
        def update_worker_status(self, rank, state):
            self.state = state
        def update_best_model(self, *args):
            raise AssertionError("Missing raw returns must never enter model ranking")
    class Callback(cfd.CallbackLogic, Base):
        def _save_checkpoint(self, *args):
            raise AssertionError("Missing raw returns must never save a ranked model")
    manager = Manager()
    callback = Callback(str(tmp_path / "worker"), 0, manager, {}, 0, {})
    callback.locals = {"rewards": [9999], "dones": [True], "infos": [{}]}
    assert callback._on_step() is False
    assert manager.state == cfd.SharedTrainingManager.STATUS_ERROR
    assert callback.episode_rewards == []
    callback._on_training_end()


def test_resume_rejects_incompatible_stage_before_solver_creation(tmp_path):
    directory = tmp_path / 'saved_models' / 'worker_0'
    directory.mkdir(parents=True)
    model, stats = directory / 'saved_model.zip', directory / 'saved_vecnormalize.pkl'
    model.write_bytes(b'model')
    stats.write_bytes(b'statistics')
    contract = cfd.cfd_contract([1, 0], 'setup:initialize')
    train.write_metadata(model, stats, {**contract, 'stage': 'rom'}, 0, 1)
    with pytest.raises(ValueError, match='stage'):
        cfd.select_resume_bundle(directory, tmp_path, contract)


def test_resume_rejects_mixed_normalization_files(tmp_path):
    model, stats = tmp_path / 'model.zip', tmp_path / 'stats.pkl'
    model.write_bytes(b'model')
    stats.write_bytes(b'matching statistics')
    contract = cfd.cfd_contract([1, 0], 'setup:initialize')
    metadata = train.write_metadata(model, stats, contract, 0, 1)
    stats.write_bytes(b'another model statistics')
    with pytest.raises(ValueError, match='integrity mismatch'):
        train.validate_bundle(model, stats, metadata, contract)


def test_automatic_rom_transfer_is_rejected_before_workers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Do not launch workers for unsupported transfer')
    monkeypatch.setattr(cfd.mp, 'Process', forbidden)
    with pytest.raises(SystemExit) as error:
        cfd.main(['--target', '1', '0', '--initializer', 'setup:initialize',
                  '--rom-checkpoint', 'model.zip'])
    assert error.value.code == 2
