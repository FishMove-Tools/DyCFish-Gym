"""PPO stop-signal tests using callback doubles, not a Fluent solver."""
from CFD_stage import training as cfd


def test_fifth_solver_failure_reaches_ppo_stop_signal(tmp_path):
    class Base:
        def __init__(self, verbose):
            self.locals = {}

    class Manager:
        def __init__(self):
            self.states = []
        def update_worker_status(self, rank, state):
            self.states.append(state)
        def update_best_model(self, *args):
            return False
        def get_best_reward(self):
            return 0.0

    class Callback(cfd.CallbackLogic, Base):
        def _save_checkpoint(self, *args):
            pass  # Do not serialize or train; exercise the actual step/stop logic.

    manager = Manager()
    callback = Callback(str(tmp_path / "worker_0"), 0, manager)
    for count in range(1, 6):
        callback.locals = {"rewards": [0.0], "dones": [True],
                           "infos": [{"failure_reason": "fluent_exception", "episode": {"r": 0.0}}]}
        assert callback._on_step() is (count < 5)
    assert callback.should_stop
    assert manager.states[-1] == cfd.SharedTrainingManager.STATUS_ERROR
    assert callback.episode_file is None


def test_non_solver_episode_breaks_consecutive_failure_count(tmp_path):
    class Base:
        def __init__(self, verbose):
            self.locals = {}
    class Manager:
        def update_worker_status(self, *args):
            pass
        def update_best_model(self, *args):
            return False
        def get_best_reward(self):
            return 0.0
    class Callback(cfd.CallbackLogic, Base):
        def _save_checkpoint(self, *args):
            pass
    callback = Callback(str(tmp_path / "worker"), 0, Manager())
    for reason in ["fluent_exception"] * 4 + ["out_of_flow_domain"] + ["fluent_exception"]:
        callback.locals = {"rewards": [0], "dones": [True], "infos": [{"failure_reason": reason, "episode": {"r": 0.0}}]}
        assert callback._on_step()
    assert callback.consecutive_failures == 1




def test_worker_does_not_overwrite_callback_failure_with_done(tmp_path, monkeypatch):
    import sys
    import types

    class Env:
        closed = False
        def seed(self, seed):
            pass
        def save(self, path):
            pass
        def close(self):
            self.closed = True

    class Model:
        def __init__(self, *args, **kwargs):
            pass
        def learn(self, **kwargs):
            return self
        def save(self, path):
            pass

    class Manager:
        states = []
        def update_worker_status(self, rank, state):
            self.states.append(state)

    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    torch.nn = types.SimpleNamespace(ReLU=object)
    sb3 = types.ModuleType('stable_baselines3')
    sb3.PPO = Model
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'stable_baselines3', sb3)
    env = Env()
    callback = types.SimpleNamespace(should_stop=True, _on_training_end=lambda: None)
    monkeypatch.setattr(cfd, 'create_callback', lambda *args, **kwargs: callback)
    monkeypatch.setattr(cfd, 'resolve_initializer', lambda spec: object())
    monkeypatch.setattr(cfd.time, 'sleep', lambda seconds: None)
    builder = 'build_env_with_optional_resume' if hasattr(cfd, 'build_env_with_optional_resume') else 'build_env'
    monkeypatch.setattr(cfd, builder, lambda *args, **kwargs: env)
    if hasattr(cfd, 'save_bundle'):
        monkeypatch.setattr(cfd, 'save_bundle', lambda *args, **kwargs: None)
    manager = Manager()
    cfd.train_with_rank(0, manager, {'output_dir': str(tmp_path), 'seed': 7,
        'target': [1, 0], 'initializer': 'setup:initialize', 'timesteps': 1, 'resume': False})
    assert manager.states[-1] == cfd.SharedTrainingManager.STATUS_ERROR
    assert cfd.SharedTrainingManager.STATUS_DONE not in manager.states
    assert env.closed
