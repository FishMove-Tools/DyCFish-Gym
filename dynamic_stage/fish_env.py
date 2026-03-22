"""Gymnasium environment for a simplified fish escape task.

This module defines :class:`FishEnv`, a lightweight 2-D rigid-body
simulation of a fish evading a pursuing predator.  Unlike the CFD stage
(which couples to ANSYS Fluent), this environment uses an analytical
dynamics model, making it suitable for rapid prototyping and large-scale
training.
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import spaces

# Limit CPU parallelism if running on a multi-core machine.
torch.set_num_threads(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FishEnv(gym.Env):
    """2-D fish escape environment with a pursuing predator.

    At the start of each episode a predator is spawned within a random
    annular region around the fish.  The predator pursues at constant
    speed.  The agent controls the fish's tail-beat amplitude and
    frequency.  The episode terminates when the fish is captured, leaves
    the domain, or survives until *max_steps*.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 2000,
        render_mode: Optional[str] = "human",
        predator_speed: float = 0.3,
        capture_radius: float = 0.1,
        render_skip: int = 10,
        enable_step_logging: bool = False,
    ) -> None:
        super().__init__()

        # General parameters
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_skip = int(render_skip)
        self.enable_step_logging = bool(enable_step_logging)

        # Fish body parameters
        self.mb: float = 1.0    # body mass (kg)
        self.Jb: float = 0.005  # body moment of inertia (kg·m²)
        self.mt: float = 0.2    # tail mass (kg)
        self.Jt: float = 0.001  # tail moment of inertia (kg·m²)
        self.r_: float = 0.3    # tail-to-body-COM distance (m)

        # Observation: [pos_x, pos_y, yaw, vbx, vby, wbz, time]
        obs_high = np.array([100.0] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(
            -obs_high, obs_high, dtype=np.float32,
        )
        # Action: [amplitude, angular_frequency]
        self.action_space = spaces.Box(
            low=np.array([0, math.pi]),
            high=np.array([math.pi / 4, math.pi * 2]),
            dtype=np.float32,
        )

        # State variables
        self.pos = np.zeros(2, dtype=np.float32)
        self.yaw: float = 0.0
        self.Vb = np.zeros(3, dtype=np.float32)  # body-frame velocity
        self.Wb = np.zeros(3, dtype=np.float32)  # body-frame angular velocity
        self.time: float = 0.0
        self.current_step: int = 0

        # Predator state
        self.predator_speed = float(predator_speed)
        self.capture_radius = float(capture_radius)
        self.predator_pos = np.zeros(2, dtype=np.float32)

        # Logging control
        self.episode_index: int = 0
        self.episode_logged: bool = False
        self._log_buffer: List[List[float]] = []

        # Rendering
        self.screen_dim: int = 600
        self.scale: float = 100.0
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

    # Gymnasium API

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and spawn a new predator position."""
        super().reset(seed=seed)

        self.pos[:] = 0.0
        self.yaw = 0.0
        self.Vb[:] = 0.0
        self.Wb[:] = 0.0
        self.time = 0.0
        self.current_step = 0
        self.episode_logged = False
        self._log_buffer = []

        # Spawn predator in a 0.5–1.0 m annular ring behind the fish
        r = random.uniform(0.5, 1.0)
        ang = random.uniform(1.5 * math.pi, 2 * math.pi)
        self.predator_pos = self.pos + r * np.array(
            [math.cos(ang), math.sin(ang)], dtype=np.float32,
        )

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the simulation by one control step.

        *action* is a two-element array ``[amplitude, angular_frequency]``.
        """
        self.A = float(action[0])
        self.w = float(action[1])

        # Advance rigid-body dynamics
        self._update_dynamics()

        # Predator pursuit
        vec_pf = self.pos - self.predator_pos
        dist_pf = float(np.linalg.norm(vec_pf))
        if dist_pf > 1e-8:
            self.predator_pos += (
                (vec_pf / dist_pf) * self.predator_speed * self.dt
            )
        else:
            # Add small jitter to break degeneracy when coincident
            jitter = (np.random.rand(2) - 0.5) * 1e-3
            self.predator_pos += jitter.astype(np.float32)
            dist_pf = 0.0

        # Reward shaping
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        vx_world = cos_yaw * self.Vb[0] - sin_yaw * self.Vb[1]
        vy_world = sin_yaw * self.Vb[0] + cos_yaw * self.Vb[1]

        unit_away = (
            (vec_pf / (dist_pf + 1e-8)) if dist_pf > 0
            else np.zeros(2, dtype=np.float32)
        )
        v_away = vx_world * unit_away[0] + vy_world * unit_away[1]

        # Reward coefficients
        r_live = 0.02
        k_dist, k_vx, k_away = 0.2, 0.3, 0.5
        k_vy, k_wz = 0.15, 0.05
        k_a_reg, k_w_reg = 0.002, 0.0005

        reward = 0.0
        reward += r_live
        reward += k_dist * min(dist_pf, 1.5)
        reward += k_vx * max(0.0, vx_world)
        reward += k_away * max(0.0, v_away)
        reward -= k_vy * abs(vy_world)
        reward -= k_wz * abs(self.Wb[2])
        reward -= k_a_reg * (self.A ** 2)
        reward -= k_w_reg * (self.w ** 2)

        terminated = False
        truncated = False
        reason: Optional[str] = None

        # Capture termination
        if dist_pf <= self.capture_radius:
            terminated = True
            reward -= 50.0
            reason = "captured"

        # Out-of-bounds termination
        if not terminated and (abs(self.pos[0]) > 50 or abs(self.pos[1]) > 50):
            terminated = True
            reward -= 10.0
            reason = "out_of_bounds"

        self.current_step += 1

        # Survival success
        if not terminated and self.current_step >= self.max_steps:
            terminated = True
            reward += 100.0
            reason = "timeout_survive"

        # Throttled rendering in human mode
        if (
            self.render_mode == "human"
            and (self.current_step % max(1, self.render_skip) == 0)
        ):
            self.render()

        # Log episode outcome (once per episode)
        if terminated and not self.episode_logged:
            success = reason == "timeout_survive"
            self._log_episode_outcome(
                success=success,
                steps=self.current_step,
                reason=reason or "none",
            )
            self._flush_step_buffer()
            self.episode_logged = True
            self.episode_index += 1

        return self._get_obs(), reward, terminated, False, {}

    def close(self) -> None:
        """Release Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    # Observation

    def _get_obs(self) -> np.ndarray:
        """Return the current observation vector."""
        return np.array(
            [
                self.pos[0], self.pos[1], self.yaw,
                self.Vb[0], self.Vb[1], self.Wb[2], self.time,
            ],
            dtype=np.float32,
        )

    # Rigid-body dynamics

    def _update_dynamics(self) -> None:
        """Integrate the fish + tail rigid-body equations for one time-step."""
        dt = self.dt
        t = self.time

        # Tail kinematics
        theta = self.A * math.cos(self.w * t)
        dtheta = -self.w * self.A * math.sin(self.w * t)
        ddtheta = -(self.w ** 2) * self.A * math.cos(self.w * t)

        r_bt = self.r_ * np.array(
            [math.cos(theta), math.sin(theta), 0.0],
        )
        Wt = self.Wb + dtheta * np.array([0, 0, 1])
        Vt = self.Vb + np.cross(Wt, r_bt)

        # Drag forces / moments on the body
        CFb = 10 * np.array([0.1, 0.01, 0])
        Fdb = -0.5 * CFb * np.sign(self.Vb) * (self.Vb ** 2)
        CMb = np.array([0, 0, 1])
        Mdb = -0.5 * CMb * np.sign(self.Wb) * (self.Wb ** 2)

        # Drag forces on the tail
        CFt = np.array([0.1, 0.1, 0.1])
        Fdt = -0.5 * CFt * np.sign(Vt) * (Vt ** 2)

        # Coupling forces and moments
        F1 = self.mt * (
            np.cross(r_bt, ddtheta * np.array([0, 0, 1]))
            - np.cross(Wt, self.Vb)
            - np.cross(Wt, np.cross(Wt, r_bt))
        )
        M1 = (
            -self.Jt * ddtheta * np.array([0, 0, 1])
            - np.cross(Wt, self.Jt * Wt)
            + np.cross(r_bt, F1)
        )

        K = np.concatenate([
            self.mb * np.cross(self.Wb, self.Vb) - F1 - Fdb,
            np.cross(self.Wb, self.Jb * self.Wb) - M1 - Mdb,
        ])

        # Construct the mass matrix and solve
        r_bt_cross = np.array([
            [0, -r_bt[2], r_bt[1]],
            [r_bt[2], 0, -r_bt[0]],
            [-r_bt[1], r_bt[0], 0],
        ])
        H = np.block([
            [
                -(self.mt + self.mb) * np.eye(3),
                self.mt * r_bt_cross,
            ],
            [
                -self.mt * r_bt_cross,
                -(self.Jt + self.Jb) * np.eye(3)
                + self.mt * (r_bt_cross @ r_bt_cross),
            ],
        ])
        # Regularise for numerical stability
        H_reg = H + 1e-8 * np.eye(6)
        U = np.linalg.solve(H_reg, K)
        dVb, dWb = U[:3], U[3:]

        # Semi-implicit Euler integration
        self.Vb += dVb * dt
        self.Wb += dWb * dt
        self.yaw += self.Wb[2] * dt

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        vx_world = cos_yaw * self.Vb[0] - sin_yaw * self.Vb[1]
        vy_world = sin_yaw * self.Vb[0] + cos_yaw * self.Vb[1]
        self.pos[0] += vx_world * dt
        self.pos[1] += vy_world * dt

        # Buffer per-step state if logging is enabled
        if self.enable_step_logging:
            self._log_buffer.append([
                self.pos[0], self.pos[1], self.yaw,
                self.Vb[0], self.Vb[1], self.Wb[2],
            ])

        self.time += dt

    # Logging utilities

    def _flush_step_buffer(self) -> None:
        """Write the accumulated per-step buffer to disk (batch I/O)."""
        if not self._log_buffer:
            return
        header_needed = not os.path.exists("varable_record.txt")
        with open(
            "varable_record.txt", "a", encoding="utf-8", newline=""
        ) as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["posx", "posy", "yaw", "Vbx", "Vby", "Wbz"])
            w.writerows(self._log_buffer)
        self._log_buffer = []

    def _log_episode_outcome(
        self, success: bool, steps: int, reason: str,
    ) -> None:
        """Append one row to the episode-outcomes CSV."""
        header_needed = not os.path.exists("episode_outcomes.csv")
        with open(
            "episode_outcomes.csv", "a", encoding="utf-8", newline=""
        ) as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["episode", "success", "steps", "reason"])
            w.writerow([
                self.episode_index, bool(success), int(steps), str(reason),
            ])

    # Rendering (Pygame)
    def render(self) -> Optional[np.ndarray]:
        """Render the current state via Pygame; returns an RGB frame in
        ``rgb_array`` mode, ``None`` otherwise.
        """
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_dim, self.screen_dim),
            )
            self.clock = pygame.time.Clock()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Reference crosshairs
        cx = self.screen_dim // 2
        cy = self.screen_dim // 2
        pygame.draw.line(
            self.screen, (200, 200, 200), (cx, 0), (cx, self.screen_dim), 1,
        )
        pygame.draw.line(
            self.screen, (200, 200, 200), (0, cy), (self.screen_dim, cy), 1,
        )

        # Fish body polygon
        body_poly = np.array(
            [[-0.25, -0.1], [0.25, -0.1], [0.25, 0.1], [-0.25, 0.1]],
            dtype=np.float32,
        )
        tail_poly = np.array(
            [[0.0, -0.05], [0.3, -0.05], [0.3, 0.05], [0.0, 0.05]],
            dtype=np.float32,
        )
        theta = self.A * math.cos(self.w * self.time)

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        R_body = np.array(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32,
        )
        body_world = (R_body @ body_poly.T).T * self.scale
        body_world[:, 0] += self.pos[0] * self.scale + cx
        body_world[:, 1] += self.pos[1] * self.scale + cy

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R_tail_local = np.array(
            [[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32,
        )
        tail_world = (R_tail_local @ tail_poly.T).T
        tail_base_body = np.array([0.25, 0.0], dtype=np.float32)
        tail_base_world = (R_body @ tail_base_body) * self.scale
        tail_world *= self.scale
        tail_world[:, 0] += tail_base_world[0] + self.pos[0] * self.scale + cx
        tail_world[:, 1] += tail_base_world[1] + self.pos[1] * self.scale + cy

        pygame.draw.polygon(
            self.screen, (0, 0, 255),
            np.round(body_world).astype(np.int32),
        )
        pygame.draw.polygon(
            self.screen, (255, 0, 0),
            np.round(tail_world).astype(np.int32),
        )

        # Predator marker
        pred_px = int(self.predator_pos[0] * self.scale + cx)
        pred_py = int(self.predator_pos[1] * self.scale + cy)
        pygame.draw.circle(self.screen, (255, 0, 0), (pred_px, pred_py), 5)
        try:
            font = pygame.font.SysFont(None, 18)
            label = font.render("predator", True, (0, 0, 0))
            self.screen.blit(label, (pred_px + 6, pred_py - 6))
        except Exception:
            pass

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, axes=(1, 0, 2))

        return None
