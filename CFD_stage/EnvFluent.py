"""Gymnasium environment wrapping ANSYS Fluent for fish locomotion control.

This module defines :class:`FluentEnv`, a Gymnasium-compatible reinforcement
learning environment that interfaces with ANSYS Fluent to simulate a
self-propelled fish in a 2-D flow domain.  The agent controls the fish's
undulation frequency and amplitude while a predator pursues it.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import ansys.fluent.core as pyfluent


class FluentEnv(gym.Env):
    """Gymnasium environment for fish escape control via ANSYS Fluent CFD.

    The fish is modelled as a self-propelled body whose kinematics are
    parameterised by undulation frequency and amplitude.  A point-mass
    predator chases the fish at a constant speed; the episode terminates
    upon capture, collision with an obstacle, domain exit, or timeout.
    """

    def __init__(
        self,
        max_steps: int = 2000,
        reward_function: str = "escape",
        simu_name: str = "CFD_0",
        predator_speed: float = 0.3,
        capture_radius: float = 0.1,
    ) -> None:
        super().__init__()
        print(f"--- Initializing FluentEnv: {simu_name} ---")

        # General parameters
        self.simu_name = simu_name
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.log_file = f"log_{simu_name}.csv"
        self.device: Optional[str] = None
        self.action_summary_file = "action_summary.txt"

        # Environment constants
        self.predator_speed = predator_speed
        self.capture_radius = capture_radius
        self.flow_domain_x_min: float = -4.0
        self.flow_domain_x_max: float = 12.0
        self.flow_domain_y_min: float = -2.0
        self.flow_domain_y_max: float = 2.0

        # Discretised physics / action parameters
        self.period_options = [2]  # available undulation period options
        self.turning_options = [
            -0.1, 0.08, -0.06, -0.04, -0.02,
            0, 0.02, 0.04, 0.06, 0.08, 0.1,
        ]  # curvature coefficient options
        self.delta_options = [-1, 0, 1]  # [at_delta, tc_delta]

        # Predator state
        self.predator_pos = np.zeros(2, dtype=np.float32)
        self._generate_predator_position()

        # Observation: [x, y, theta, vx, vy, wz, time]
        self.observation_space = spaces.Box(
            low=np.array([
                self.flow_domain_x_min, self.flow_domain_y_min,
                -np.pi, -np.inf, -np.inf, -np.inf, 0.0,
            ]),
            high=np.array([
                self.flow_domain_x_max, self.flow_domain_y_max,
                np.pi, np.inf, np.inf, np.inf, np.inf,
            ]),
            dtype=np.float64,
        )
        # Action: [frequency, amplitude] -> mapped to [period, curvature]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([2.0, np.pi / 4]),
            dtype=np.float32,
        )

        # Episode state variables
        self.episode_number: int = 0
        self.current_step: int = 0
        self.simulation_time: float = 0.0
        self.time_step: float = 0.01  # Fluent simulation time-step (s)
        self.fish_position = np.array([0.0, 0.0])
        self.fish_orientation: float = 0.0
        self.state = np.zeros(6, dtype=np.float64)

        # Working directory
        self.env_dir = os.path.join("fishmove", f"{self.simu_name}")
        os.makedirs(self.env_dir, exist_ok=True)
        os.chdir(self.env_dir)

        # Fluent console transcript
        self.console_log = "fluent_console.log"
        self._transcript_active = False

        # Launch Fluent solver
        self.solver = pyfluent.launch_fluent(
            precision="double",
            processor_count=6,
            dimension=2,
            ui_mode="gui",  # gui | no_gui_or_graphics | no_gui
        )
        self.start_class(complete_reset=True)
        print(f"--- FluentEnv {self.simu_name} initialized ---")

    # Internal helpers

    def _generate_predator_position(self) -> None:
        """Randomly place the predator 0.5–1.0 m from the fish."""
        r = random.uniform(0.5, 1.0)
        ang = random.uniform(1.5 * np.pi, 2 * np.pi)
        self.predator_pos = self.fish_position + r * np.array(
            [np.cos(ang), np.sin(ang)], dtype=np.float32,
        )

    # Gymnasium API

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.initialize_flow(complete_reset=True)
        self.current_step = 0
        self.episode_number += 1
        self.simulation_time = 0.0
        self.fish_position = np.array([0.0, 0.0])
        self.fish_orientation = 0.0
        self.state = np.zeros(6, dtype=np.float64)
        self._generate_predator_position()
        self.prev_target_distance = np.linalg.norm(
            self.fish_position - self.target_position
        )
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one control action and advance the Fluent simulation.

        *action* is a two-element array ``[frequency, amplitude]``.
        """
        # 1) Decode control action [frequency, amplitude]
        frequency = float(action[0])
        amplitude = float(action[1])

        # 2) Map to period and curvature coefficient
        period = 1.0 / frequency
        a_s = (amplitude / (np.pi / 4)) * 1.4
        curvature = a_s * 0.1  # mapped to [-0.1, 0.1] range

        # 3) Update control parameters
        self.current_period_value = period
        self.current_turning_value = curvature
        steps_to_execute = int(self.current_period_value / self.time_step)

        # 4) Status flags
        failed = False
        success = False
        failure_reason = ""
        terminated = False
        truncated = False
        step_i = -1

        # 5) Synchronise parameters with Fluent
        self.solver.execute_tui(f"/solve/set/time-step {self.time_step}")
        self.solver.execute_tui(f"(rpsetvar 'tc {self.current_period_value})")
        self.solver.execute_tui(f"(rpsetvar 'at {self.current_turning_value})")

        # 6) Advance simulation in Fluent
        for step_i in range(steps_to_execute):
            if step_i == 0:
                self.solver.execute_tui(
                    '/define/user-defined/execute-on-demand '
                    '"add_action_from_console::libudf"'
                )

            if self.current_step >= self.max_steps:
                truncated = True
                break

            try:
                self.simulation_time += self.time_step
                self.current_step += 1
                self.solver.execute_tui("/solve/dual-time-iterate 1 10")

                # Update fish pose
                xdisp, ydisp, thetadisp, _, _, _ = self._read_output_file()
                self.fish_position[0] = xdisp
                self.fish_position[1] = ydisp
                self.fish_orientation = thetadisp

                # Predator pursuit behaviour
                vec_pf = self.fish_position - self.predator_pos
                dist_pf = float(np.linalg.norm(vec_pf))
                if dist_pf > 1e-8:
                    self.predator_pos += (
                        (vec_pf / dist_pf) * self.predator_speed * self.time_step
                    )
                else:
                    jitter = (np.random.rand(2) - 0.5) * 1e-3
                    self.predator_pos += jitter.astype(np.float32)

                # Collision / out-of-domain checks
                obstacle_distance = self._calculate_obstacle_distance()
                if obstacle_distance < (self.obstacle_diameter / 2 + 0.02):
                    failed = True
                    failure_reason = "collision_with_obstacle"
                    terminated = True
                    break

                if (
                    self.fish_position[0] > self.flow_domain_x_max
                    or self.fish_position[0] < self.flow_domain_x_min
                    or self.fish_position[1] > self.flow_domain_y_max
                    or self.fish_position[1] < self.flow_domain_y_min
                ):
                    failed = True
                    failure_reason = "out_of_flow_domain"
                    terminated = True
                    break

            except Exception as e:
                print(
                    f"[{self.simu_name}] Error in simulation step {step_i}: {e}"
                )
                failed = True
                failure_reason = "fluent_exception"
                terminated = True
                break

        # Compute reward
        self.state = self._get_obs()
        target_distance = np.linalg.norm(
            self.fish_position - self.target_position
        )
        success = target_distance < 0.2
        if success:
            terminated = True

        reward = -target_distance * 10.0  # distance-based penalty
        success_reward = 1000 if success else 0

        if failed:
            if failure_reason == "collision_with_obstacle":
                reward -= 500
            elif failure_reason == "out_of_flow_domain":
                reward -= 400
            elif failure_reason == "fluent_exception":
                reward -= 1000

        info: Dict[str, Any] = {
            "simulation_time": self.simulation_time,
            "turning_action": self.current_turning_value,
            "period_action": self.current_period_value,
            "fish_position": self.fish_position.copy(),
            "fish_orientation": self.fish_orientation,
            "obstacle_distance": obstacle_distance,
            "target_distance": target_distance,
            "success": success,
            "failed": failed,
            "failure_reason": failure_reason,
            "timeout": truncated,
            "steps_executed": (step_i + 1) if step_i >= 0 else 0,
        }

        return self.state.copy(), reward + success_reward, terminated, truncated, info

    def close(self) -> None:
        """Shut down the Fluent solver and clean up resources."""
        if hasattr(self, "solver") and self.solver is not None:
            try:
                self._stop_transcript_safe()
                self.solver.exit()
            except Exception as e:
                print(f"[{self.simu_name}] Error closing Fluent: {e}")

    # Logging

    def _log_variables(self) -> None:
        """Append the current simulation state to the variable record CSV."""
        filename = "variable_record.txt"
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode, encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            if mode == "w":
                header = [
                    "simulation_time",
                    "x_disp",
                    "y_disp",
                    "theta_disp",
                    "turning_action",
                    "period_action",
                    "obstacle_distance",
                    "target_distance",
                ]
                writer.writerow(header)
            writer.writerow([
                self.simulation_time,
                self.fish_position[0],
                self.fish_position[1],
                self.fish_orientation,
                self.current_turning_value,
                self.current_period_value,
                self._calculate_obstacle_distance(),
                np.linalg.norm(self.fish_position - self.target_position),
            ])
