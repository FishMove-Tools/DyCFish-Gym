"""Explicit target-navigation wrapper for the supplied Fluent UDF.

The archived CFD reward is a target reward, not a reconstructed escape task.
Only ``interface_profile='legacy'`` is supported here. The manuscript escape
interface requires a separately validated CFD task, action mapping and reset
procedure. See ``fluent_backend.py`` for the real case initialization contract.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .fluent_backend import FluentBackend
    from .interface import action_bounds, decode_action, observation
except ImportError:  # Historical direct script import.
    from fluent_backend import FluentBackend
    from interface import action_bounds, decode_action, observation


class FluentEnv(gym.Env):
    """Legacy CFD target task with explicit physical configuration.

    ``target_position`` and ``initializer(solver, work_dir)`` are required.
    The initializer must fully reset real flow and UDF globals, then return the
    actual initial ``[x,y,yaw,vx_world,vy_world,wz,time]`` state. The archive does
    not supply this initialization journal. Injecting a solver supports tests
    and pre-existing sessions; it does not remove the initialization requirement.

    Legacy actions are ``[frequency_hz, amplitude]``; each lasts up to one period
    or the remaining CFD-step budget. The diagnostic predator does not determine
    this target task's reward, success or termination. Escape is unsupported.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 2000,
        reward_function: str = "target",
        simu_name: str = "CFD_0",
        predator_speed: float = 0.3,
        capture_radius: float = 0.1,
        *,
        target_position=None,
        obstacle_position=None,
        obstacle_diameter: float = 0.0,
        interface_profile: str = "legacy",
        time_step: float = 0.01,
        work_dir=None,
        solver=None,
        initializer: Callable | None = None,
        launch_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if interface_profile != "legacy" or reward_function != "target":
            raise NotImplementedError(
                "CFD supports only interface_profile='legacy', reward_function='target'. "
                "The manuscript_escape CFD task has not been recovered or validated."
            )
        if target_position is None:
            raise ValueError("target_position must be explicitly supplied; no target is inferred.")
        if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive.")
        if not np.isfinite(predator_speed) or predator_speed < 0:
            raise ValueError("predator_speed must be finite and nonnegative.")
        if not np.isfinite(capture_radius) or capture_radius <= 0:
            raise ValueError("capture_radius must be finite and positive (diagnostic only).")
        if Path(simu_name).name != simu_name or simu_name in {"", ".", ".."}:
            raise ValueError("simu_name must be a single directory name.")
        self.interface_profile = interface_profile
        self.reward_function = reward_function
        self.simu_name = simu_name
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.predator_speed = float(predator_speed)
        self.capture_radius = float(capture_radius)
        self.flow_domain_x_min, self.flow_domain_x_max = -4.0, 12.0
        self.flow_domain_y_min, self.flow_domain_y_max = -2.0, 2.0
        self.target_position = self._position(target_position, "target_position")
        if self._outside_domain(self.target_position):
            raise ValueError("target_position must lie inside the configured [-4,12] x [-2,2] domain.")
        self.obstacle_position = (
            None if obstacle_position is None else self._position(obstacle_position, "obstacle_position")
        )
        if not np.isfinite(obstacle_diameter) or obstacle_diameter < 0:
            raise ValueError("obstacle_diameter must be finite and nonnegative.")
        if self.obstacle_position is not None and obstacle_diameter <= 0:
            raise ValueError("A configured obstacle requires a positive obstacle_diameter.")
        if self.obstacle_position is None and obstacle_diameter != 0:
            raise ValueError("obstacle_position is required when obstacle_diameter is nonzero.")
        self.obstacle_diameter = float(obstacle_diameter)

        low, high = action_bounds()
        self.action_space = spaces.Box(np.asarray(low, dtype=np.float32),
                                       np.asarray(high, dtype=np.float32), dtype=np.float32)
        # Terminal states may lie outside the domain; the UDF yaw is unwrapped.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 6 + [0.0], dtype=np.float64),
            high=np.full(7, np.inf, dtype=np.float64), dtype=np.float64,
        )
        self.env_dir = Path(work_dir or Path.cwd() / "runs" / simu_name).resolve()
        self.backend = FluentBackend(self.env_dir, solver=solver,
                                     initializer=initializer, launch_kwargs=launch_kwargs)
        self.log_file = self.env_dir / "environment_steps.csv"
        self.console_log = self.env_dir / "fluent_console.log"
        self.render_mode = None
        self.episode_number = 0
        self.current_step = 0
        self.simulation_time = 0.0
        self.fish_position = np.zeros(2, dtype=np.float64)
        self.fish_orientation = 0.0
        self.predator_pos = np.zeros(2, dtype=np.float64)
        self.state = np.zeros(7, dtype=np.float64)
        self.current_period_value = 0.0
        self.current_turning_value = 0.0
        self._needs_reset = True
        self._closed = False

    @property
    def solver(self):
        return self.backend.solver

    @staticmethod
    def _position(value, name):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain two finite coordinates.")
        return result.copy()

    def _outside_domain(self, position):
        x, y = position
        return (x < self.flow_domain_x_min or x > self.flow_domain_x_max or
                y < self.flow_domain_y_min or y > self.flow_domain_y_max)

    def _generate_predator_position(self):
        radius = self.np_random.uniform(0.5, 1.0)
        angle = self.np_random.uniform(1.5 * np.pi, 2.0 * np.pi)
        self.predator_pos = self.fish_position + radius * np.array([np.cos(angle), np.sin(angle)])

    def _calculate_obstacle_distance(self):
        if self.obstacle_position is None:
            return float("inf")
        return float(np.linalg.norm(self.fish_position - self.obstacle_position))

    def _get_obs(self):
        return np.asarray(observation(self.state), dtype=np.float64)

    def _set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.fish_position = self.state[:2].copy()
        self.fish_orientation = float(self.state[2])
        self.simulation_time = float(self.state[6])

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Cannot reset a closed FluentEnv.")
        super().reset(seed=seed)
        self._needs_reset = True
        initial_state = self.backend.reset()
        self._set_state(initial_state)
        if self._outside_domain(self.fish_position):
            raise ValueError("The initialized fish position is outside the CFD domain.")
        self.current_step = 0
        self.current_period_value = 0.0
        self.current_turning_value = 0.0
        self.episode_number += 1
        self._generate_predator_position()
        self._needs_reset = False
        return self._get_obs(), {"task": "target", "interface_profile": self.interface_profile,
                                 "predator_position": self.predator_pos.copy()}

    def step(self, action):
        if self._closed or self._needs_reset:
            raise RuntimeError("Call reset() before stepping, including after an episode ends.")
        amplitude, frequency_hz = decode_action(action)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("CFD frequency must be strictly positive.")
        period = 1.0 / frequency_hz
        if not np.isfinite(period):
            raise ValueError("CFD frequency is too small to represent a finite period.")
        self.current_period_value = period
        # Preserve the archive: at is [0,0.14], scaling the UDF's turning term.
        # Its separate A_w oscillatory amplitude remains fixed in the C source.
        self.current_turning_value = (amplitude / (np.pi / 4)) * 0.14
        remaining = self.max_steps - self.current_step
        steps_to_execute = (remaining if period >= remaining * self.time_step
                            else max(1, int(period / self.time_step)))
        failed = success = terminated = False
        failure_reason = ""
        error_message = ""
        executed = 0
        obstacle_distance = self._calculate_obstacle_distance()
        try:
            self.backend.set_action(period, self.current_turning_value, self.time_step)
            for _ in range(steps_to_execute):
                self._set_state(self.backend.advance(self.time_step))
                self.current_step += 1
                executed += 1
                toward_fish = self.fish_position - self.predator_pos
                distance = float(np.linalg.norm(toward_fish))
                if distance > 0:
                    self.predator_pos += toward_fish / distance * min(distance, self.predator_speed * self.time_step)
                obstacle_distance = self._calculate_obstacle_distance()
                if obstacle_distance < self.obstacle_diameter / 2 + 0.02:
                    failed, terminated, failure_reason = True, True, "collision_with_obstacle"
                    break
                if self._outside_domain(self.fish_position):
                    failed, terminated, failure_reason = True, True, "out_of_flow_domain"
                    break
                if np.linalg.norm(self.fish_position - self.target_position) < 0.2:
                    success = terminated = True
                    break
        except Exception as exc:
            failed, terminated, failure_reason = True, True, "fluent_exception"
            error_message = f"{type(exc).__name__}: {exc}"

        truncated = not terminated and self.current_step >= self.max_steps
        self._needs_reset = terminated or truncated
        target_distance = float(np.linalg.norm(self.fish_position - self.target_position))
        reward = -10.0 * target_distance
        if success:
            reward += 1000.0
        if failed:
            reward -= {"collision_with_obstacle": 500.0, "out_of_flow_domain": 400.0,
                       "fluent_exception": 1000.0}[failure_reason]
        info: dict[str, Any] = {
            "task": "target", "interface_profile": self.interface_profile,
            "simulation_time": self.simulation_time, "turning_action": self.current_turning_value,
            "period_action": period, "fish_position": self.fish_position.copy(),
            "fish_orientation": self.fish_orientation, "obstacle_distance": obstacle_distance,
            "target_distance": target_distance, "success": success, "failed": failed,
            "failure_reason": failure_reason, "timeout": truncated, "steps_executed": executed,
            "error": error_message,
        }
        try:
            self._log_variables(info, reward)
        except OSError as exc:
            info["logging_error"] = f"{type(exc).__name__}: {exc}"
        return self._get_obs(), float(reward), terminated, truncated, info

    def _log_variables(self, info, reward):
        self.env_dir.mkdir(parents=True, exist_ok=True)
        header_needed = not self.log_file.exists() or self.log_file.stat().st_size == 0
        with self.log_file.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if header_needed:
                writer.writerow(["episode", "cfd_steps", "time", "x", "y", "yaw", "vx_world",
                                 "vy_world", "wz", "period", "turning", "target_distance",
                                 "obstacle_distance", "raw_reward", "success", "failed", "timeout", "error"])
            writer.writerow([self.episode_number, self.current_step, self.simulation_time,
                             *self.state[:6], self.current_period_value, self.current_turning_value,
                             info["target_distance"], info["obstacle_distance"], reward,
                             info["success"], info["failed"], info["timeout"], info["error"]])

    def close(self):
        if not self._closed:
            self._closed = True
            self._needs_reset = True
            self.backend.close()
