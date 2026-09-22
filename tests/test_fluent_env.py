"""Interface tests with an injected file-producing solver double, not CFD validation."""

from pathlib import Path

import numpy as np
import pytest

from CFD_stage.EnvFluent import FluentEnv
from CFD_stage.fluent_backend import FluentConfigurationError


class FileSolverDouble:
    """Reproduce the supplied UDF's file contract without simulating physics."""

    def __init__(self):
        self.commands = []
        self.exit_count = 0
        self.next_x = 0.01
        self.next_y = 0.0
        self.failure_prefix = None
        self.write_output = True
        self.write_snapshot = True
        self.invalid_state = False

    def initialize(self, solver, work_dir):
        assert solver is self
        self.work_dir = Path(work_dir)
        self.time = 0.0
        self.dt = 0.01
        self.state = np.zeros(7)
        return self.state.copy()

    def execute_tui(self, command):
        self.commands.append(command)
        if self.failure_prefix and command.startswith(self.failure_prefix):
            raise RuntimeError("injected solver failure")
        if command.startswith("/solve/set/time-step "):
            self.dt = float(command.rsplit(" ", 1)[1])
        elif command.startswith("/solve/dual-time-iterate"):
            self.time += self.dt
            self.state = np.array([self.next_x, self.next_y, 0.2, 0.4, -0.2, 0.3, self.time])
            if self.write_output:
                with (self.work_dir / "Output.txt").open("a", encoding="ascii") as stream:
                    # Force columns are deliberately unlike the actual velocity.
                    stream.write(f"{self.time:.4e} {self.next_x:.4e} {self.next_y:.4e} 2.0000e-1 123 456 789 0\n")
        elif "save_fish_state::libudf" in command and self.write_snapshot:
            filename = f"fish_state_t{self.time:.6f}.txt"
            values = self.state[:6].copy()
            if self.invalid_state:
                values[3] = np.nan
            names = ("XDISP", "YDISP", "THETADISP", "XVEL", "YVEL", "THETAVEL")
            with (self.work_dir / filename).open("w", encoding="ascii") as stream:
                for name, value in zip(names, values):
                    stream.write(f"{name} {value:.16e}\n")
                stream.write("SENSOR_COORDS_X 0 0.2 0.4\n")
            (self.work_dir / "fish_latest_state.txt").write_text(filename + "\n", encoding="ascii")

    def exit(self):
        self.exit_count += 1


@pytest.fixture
def configured(tmp_path):
    solver = FileSolverDouble()
    env = FluentEnv(target_position=(2.0, 0.0), solver=solver,
                    initializer=solver.initialize, work_dir=tmp_path, max_steps=2)
    yield env, solver
    env.close()


def test_configuration_is_explicit_and_escape_is_not_misrepresented():
    with pytest.raises(ValueError, match="target_position"):
        FluentEnv()
    with pytest.raises(FluentConfigurationError, match="initializer"):
        FluentEnv(target_position=(2, 0))
    with pytest.raises(ValueError, match="inside"):
        FluentEnv(target_position=(-5, 0))
    with pytest.raises(NotImplementedError, match="not been recovered"):
        FluentEnv(target_position=(2, 0), reward_function="escape")
    with pytest.raises(NotImplementedError, match="not been recovered"):
        FluentEnv(target_position=(2, 0), interface_profile="manuscript_escape")


def test_reset_is_seven_dimensional_seeded_and_does_not_change_cwd(configured):
    env, solver = configured
    cwd = Path.cwd()
    obs, info = env.reset(seed=123)
    assert obs.shape == (7,)
    assert env.observation_space.contains(obs)
    first_predator = info["predator_position"]
    obs, info = env.reset(seed=123)
    np.testing.assert_array_equal(first_predator, info["predator_position"])
    assert info["task"] == "target"
    assert Path.cwd() == cwd
    assert solver.commands == []  # Initialization contract supplied by this fixture.


@pytest.mark.parametrize("action", [[0, 0.1], [-1, 0.1], [1, -0.1], [3, 0.1], [1, np.nan], [1]])
def test_invalid_actions_fail_before_any_solver_command(configured, action):
    env, solver = configured
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(action)
    assert solver.commands == []


def test_steps_use_udf_velocities_and_timeout_exactly_at_budget(configured):
    env, solver = configured
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step([2, np.pi / 4])
    np.testing.assert_allclose(obs[3:6], [0.4, -0.2, 0.3], rtol=1e-6)
    assert not terminated and truncated
    assert info["steps_executed"] == 2 and env.current_step == 2
    assert info["simulation_time"] == pytest.approx(0.02)
    assert info["turning_action"] == pytest.approx(0.14)
    assert info["period_action"] == pytest.approx(0.5)
    assert reward == pytest.approx(-19.9)
    assert env.log_file.parent == env.env_dir
    assert "raw_reward" in env.log_file.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="reset"):
        env.step([2, 0.1])


def test_success_uses_explicit_target_and_stops_at_first_reaching_substep(configured):
    env, solver = configured
    solver.next_x = 1.9
    env.reset()
    obs, reward, terminated, truncated, info = env.step([2, 0.1])
    assert terminated and not truncated and info["success"] and not info["failed"]
    assert reward == pytest.approx(999.0)
    assert info["steps_executed"] == 1


@pytest.mark.parametrize("failure_prefix", ["/solve/set/time-step", "/solve/dual-time-iterate"])
def test_solver_failure_returns_finite_last_state_and_failure_penalty(configured, failure_prefix):
    env, solver = configured
    env.reset()
    solver.failure_prefix = failure_prefix
    obs, reward, terminated, truncated, info = env.step([2, 0.1])
    assert terminated and not truncated and info["failed"] and not info["success"]
    assert info["failure_reason"] == "fluent_exception"
    assert "injected solver failure" in info["error"]
    assert info["steps_executed"] == 0 and np.isfinite(obs).all()
    assert reward == pytest.approx(-1020.0)


def test_stale_output_is_not_reused_after_reset(configured):
    env, solver = configured
    env.reset()
    env.step([2, 0.1])
    env.reset()
    solver.write_output = False
    _, _, terminated, _, info = env.step([2, 0.1])
    assert terminated and info["failed"]
    assert "fresh complete" in info["error"]


def test_stale_snapshot_marker_is_not_reused_after_reset(configured):
    env, solver = configured
    env.reset()
    env.step([2, 0.1])
    env.reset()
    solver.write_snapshot = False
    _, _, terminated, _, info = env.step([2, 0.1])
    assert terminated and info["failed"]
    assert "FileNotFoundError" in info["error"]


def test_nonfinite_state_is_a_solver_failure_not_a_policy_observation(configured):
    env, solver = configured
    env.reset()
    solver.invalid_state = True
    obs, _, terminated, _, info = env.step([2, 0.1])
    assert terminated and info["failed"] and np.isfinite(obs).all()
    assert "non-finite" in info["error"]


def test_collision_takes_precedence_over_target_success(tmp_path):
    solver = FileSolverDouble()
    solver.next_x = 1.9
    env = FluentEnv(target_position=(2, 0), obstacle_position=(1.9, 0), obstacle_diameter=0.1,
                    solver=solver, initializer=solver.initialize, work_dir=tmp_path)
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step([2, 0.1])
        assert terminated and not truncated and not info["success"]
        assert info["failure_reason"] == "collision_with_obstacle"
        assert reward == pytest.approx(-501.0)
    finally:
        env.close()


def test_domain_exit_has_its_own_penalty(configured):
    env, solver = configured
    solver.next_x = 13.0
    env.reset()
    obs, reward, terminated, _, info = env.step([2, 0.1])
    assert terminated and info["failure_reason"] == "out_of_flow_domain"
    assert reward == pytest.approx(-510.0)
    assert env.observation_space.contains(obs)


def test_initial_state_contract_is_checked(tmp_path):
    solver = FileSolverDouble()
    env = FluentEnv(target_position=(2, 0), solver=solver,
                    initializer=lambda *_: np.zeros(6), work_dir=tmp_path)
    try:
        with pytest.raises(FluentConfigurationError, match="seven|finite"):
            env.reset()
        with pytest.raises(RuntimeError, match="reset"):
            env.step([2, 0.1])
    finally:
        env.close()


def test_close_is_idempotent(configured):
    env, solver = configured
    env.close()
    env.close()
    assert solver.exit_count == 1


def test_udf_fixed_action_capacity_is_enforced_and_resettable(configured):
    env, solver = configured
    env.reset()
    for _ in range(40):
        env.backend.set_action(0.5, 0.1, 0.01)
    command_count = len(solver.commands)
    with pytest.raises(RuntimeError, match="40 actions"):
        env.backend.set_action(0.5, 0.1, 0.01)
    assert len(solver.commands) == command_count
    env.reset()
    env.backend.set_action(0.5, 0.1, 0.01)


def test_logging_failure_is_reported_without_changing_physical_outcome(configured, monkeypatch):
    env, solver = configured
    env.reset()

    def unwritable(*_):
        raise OSError("injected log error")

    monkeypatch.setattr(env, "_log_variables", unwritable)
    _, reward, terminated, truncated, info = env.step([2, 0.1])
    assert not terminated and truncated and not info["failed"]
    assert reward == pytest.approx(-19.9)
    assert "injected log error" in info["logging_error"]
