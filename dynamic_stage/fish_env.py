"""Fish escape ROM aligned with manuscript Eqs. (2), (8)-(21).

Default action/observation/reward interfaces retain the supplied training setup.
See model_notes.txt for parameter provenance and remaining manuscript ambiguities.
Dependencies: numpy, gymnasium; pygame is only needed for rendering.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FishEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, dt=0.01, max_steps=2000, render_mode="human",
                 predator_speed=0.3, capture_radius=0.1, render_skip=10,
                 enable_step_logging=False, *, mb=1.0, mt=0.2,
                 Jb=0.005, Jt=0.001, r_t=0.3,
                 mb0=None, mt0=None, kb=None, kt=None,
                 action_mode="legacy_omega", log_dir="."):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render_mode")
        if action_mode not in ("legacy_omega", "paper_hz"):
            raise ValueError("action_mode must be legacy_omega or paper_hz")
        positive = (dt, mb, mt, Jb, Jt, r_t, capture_radius)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("dt, masses, inertias, r_t and capture_radius must be positive")
        if int(max_steps) != max_steps or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not np.isfinite(predator_speed) or predator_speed < 0:
            raise ValueError("predator_speed must be finite and nonnegative")
        supplied = (mb0, mt0, kb, kt)
        if any(x is not None for x in supplied):
            if any(x is None for x in supplied):
                raise ValueError("Supply mb0, mt0, kb and kt together")
            if not all(np.isfinite(x) and x > 0 for x in supplied):
                raise ValueError("Eq. (8) requires positive actual masses and kb, kt")
            mb, mt = (1 + kb) * mb0, (1 + kt) * mt0
            if not np.isfinite(mb + mt):
                raise ValueError("Equivalent masses overflowed")
        self.mb0, self.mt0, self.kb, self.kt = supplied
        # Eq. (8): mb and mt are EQUIVALENT masses, not necessarily dry masses.
        self.mb, self.mt = float(mb), float(mt)
        self.Jb, self.Jt, self.r_ = float(Jb), float(Jt), float(r_t)
        self.dt, self.max_steps = float(dt), int(max_steps)
        self.render_mode, self.render_skip = render_mode, max(1, int(render_skip))
        self.predator_speed, self.capture_radius = float(predator_speed), float(capture_radius)
        self.enable_step_logging = bool(enable_step_logging)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.action_mode = action_mode
        # Table 1, Eqs. (9)-(10): lumped loads; no extra tail drag.
        self.CFb = np.array([0.10, 0.13, 0.08], dtype=np.float64)
        self.CMb = np.array([0.20, 0.23, 0.28], dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        if action_mode == "legacy_omega":
            low, high = [0, math.pi], [math.pi / 4, 2 * math.pi]
        else:
            low, high = [-math.pi / 4, 1], [math.pi / 4, 2]
        self.action_space = spaces.Box(np.array(low, dtype=np.float32),
                                       np.array(high, dtype=np.float32))
        self.pos = np.zeros(2, dtype=np.float64)
        self.Vb = np.zeros(3, dtype=np.float64)
        self.Wb = np.zeros(3, dtype=np.float64)
        self.predator_pos = np.zeros(2, dtype=np.float64)
        self.yaw = self.time = self.A = 0.0
        self.w = math.pi if action_mode == "legacy_omega" else 2 * math.pi
        self.current_step = self.episode_index = 0
        self.episode_logged = False
        self._done = False
        self._log_buffer = []
        self.screen_dim, self.scale = 600, 100.0
        self.screen = self.clock = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos.fill(0)
        self.Vb.fill(0)
        self.Wb.fill(0)
        self.yaw = self.time = self.A = 0.0
        self.w = math.pi if self.action_mode == "legacy_omega" else 2 * math.pi
        self.current_step = 0
        self.episode_logged = self._done = False
        self._log_buffer = []
        # Preserve the original fourth-quadrant spawn sector (not a full annulus).
        radius = self.np_random.uniform(0.5, 1.0)
        angle = self.np_random.uniform(1.5 * math.pi, 2 * math.pi)
        self.predator_pos[:] = self.pos + radius * np.array([math.cos(angle), math.sin(angle)])
        return self._get_obs(), {}

    def _tail_kinematics(self):
        """Eq. (2), with A and omega held constant within each control step.

        Literal omega*t convention: changes of action may cause jumps. The
        manuscript does not specify action-transition smoothing or phase memory.
        """
        phase = self.w * self.time
        theta = self.A * math.sin(phase)
        dtheta = self.A * self.w * math.cos(phase)
        ddtheta = -self.A * self.w**2 * math.sin(phase)
        return theta, dtheta, ddtheta

    def _dynamics_system(self):
        """Positive planar mass matrix derived directly from Eqs. (17)-(21).

        Unknown = [dVbx, dVby, dWbz]. With r=(rx,ry), p=e3 x r,
        a_t = dVb + p*dWbz + b,
        b = Omega_b x Vb + ddtheta*(e3 x r) - (Wbz+dtheta)^2*r.
        """
        theta, dtheta, ddtheta = self._tail_kinematics()
        r = self.r_ * np.array([math.cos(theta), math.sin(theta)])
        p = np.array([-r[1], r[0]])
        u, v = self.Vb[:2]
        omega = self.Wb[2]
        transport = omega * np.array([-v, u])  # Omega_b, NOT Omega_t
        b = transport + ddtheta * p - (omega + dtheta)**2 * r
        drag = -0.5 * self.CFb[:2] * np.abs(self.Vb[:2]) * self.Vb[:2]
        moment = -0.5 * self.CMb[2] * abs(omega) * omega
        mass = np.array([
            [self.mb + self.mt, 0, self.mt * p[0]],
            [0, self.mb + self.mt, self.mt * p[1]],
            [self.mt * p[0], self.mt * p[1], self.Jb + self.Jt + self.mt * self.r_**2],
        ])
        rhs = np.r_[drag - self.mb * transport - self.mt * b,
                    moment - self.Jt * ddtheta - self.mt * np.dot(p, b)]
        return mass, rhs

    def _update_dynamics(self):
        mass, rhs = self._dynamics_system()
        acceleration = np.linalg.solve(mass, rhs)
        # Eqs. (3)-(7), semi-implicit Euler; double precision internally.
        self.Vb[:2] += acceleration[:2] * self.dt
        self.Wb[2] += acceleration[2] * self.dt
        self.Vb[2] = self.Wb[0] = self.Wb[1] = 0.0
        self.yaw += self.Wb[2] * self.dt
        self.pos += self._world_velocity() * self.dt
        self.time += self.dt
        if not np.all(np.isfinite(np.r_[self.pos, self.yaw, self.Vb, self.Wb])):
            raise FloatingPointError("Nonfinite ROM state; check parameters and dt")
        if self.enable_step_logging:
            self._log_buffer.append([*self.pos, self.yaw, *self.Vb[:2], self.Wb[2]])

    def _world_velocity(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * self.Vb[0] - s * self.Vb[1],
                         s * self.Vb[0] + c * self.Vb[1]])

    def _get_obs(self):
        return np.array([*self.pos, self.yaw, *self.Vb[:2], self.Wb[2], self.time], dtype=np.float32)

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is finished; call reset()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain two finite numbers")
        # Permit float32 endpoint rounding only; do not silently clip invalid actions.
        low = self.action_space.low.astype(np.float64)
        high = self.action_space.high.astype(np.float64)
        if np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
            raise ValueError("Action is outside action_space")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.A = float(action[0])
        self.w = float(action[1]) * (2 * math.pi if self.action_mode == "paper_hz" else 1)
        self._update_dynamics()
        vec = self.pos - self.predator_pos
        distance = float(np.linalg.norm(vec))
        if distance > 0:
            self.predator_pos += vec / distance * min(distance, self.predator_speed * self.dt)
        # Evaluate capture and reward at the returned state, after predator motion.
        vec = self.pos - self.predator_pos
        distance = float(np.linalg.norm(vec))
        away = vec / distance if distance > 0 else np.zeros(2)
        velocity = self._world_velocity()
        reward = (0.02 + 0.2 * min(distance, 1.5)
                  + 0.3 * max(0.0, velocity[0]) + 0.5 * max(0.0, float(velocity @ away))
                  - 0.15 * abs(velocity[1]) - 0.05 * abs(self.Wb[2])
                  - 0.002 * self.A**2 - 0.0005 * self.w**2)
        self.current_step += 1
        reason = None
        if distance <= self.capture_radius:
            reason, reward = "captured", reward - 50.0
        elif np.any(np.abs(self.pos) > 50):
            reason, reward = "out_of_bounds", reward - 10.0
        elif self.current_step >= self.max_steps:
            # Preserved finite-horizon survival objective, with time observed.
            reason, reward = "timeout_survive", reward + 100.0
        terminated = reason is not None
        self._done = terminated
        if self.render_mode == "human" and self.current_step % self.render_skip == 0:
            self.render()
        if terminated and not self.episode_logged:
            self._log_episode_outcome(reason == "timeout_survive", self.current_step, reason)
            self._flush_step_buffer()
            self.episode_logged = True
            self.episode_index += 1
        return self._get_obs(), float(reward), terminated, False, {
            "reason": reason, "predator_distance": distance,
            "predator_pos": self.predator_pos.copy(),
        }

    def _write_rows(self, filename, header, rows):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / filename
        header_needed = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(header)
            writer.writerows(rows)

    def _flush_step_buffer(self):
        if self._log_buffer:
            self._write_rows("varable_record.txt", ["posx", "posy", "yaw", "Vbx", "Vby", "Wbz"], self._log_buffer)
            self._log_buffer = []

    def _log_episode_outcome(self, success, steps, reason):
        self._write_rows("episode_outcomes.csv", ["episode", "success", "steps", "reason"],
                         [[self.episode_index, bool(success), int(steps), reason]])

    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        self.screen.fill((255, 255, 255))
        center = np.array([self.screen_dim / 2, self.screen_dim / 2])
        def pixels(points):
            return np.rint(center + np.asarray(points) * [self.scale, -self.scale]).astype(int)
        def rotation(angle):
            return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        rb = rotation(self.yaw)
        theta, _, _ = self._tail_kinematics()
        # Schematic only: modeled origin is also the hinge in Eqs. (15)-(17).
        body = np.array([[-0.5, -0.10], [0, -0.10], [0, 0.10], [-0.5, 0.10]])
        tail = np.array([[0, -0.04], [2*self.r_, -0.04], [2*self.r_, 0.04], [0, 0.04]])
        pygame.draw.polygon(self.screen, (35, 105, 210), pixels(body @ rb.T + self.pos))
        pygame.draw.polygon(self.screen, (235, 135, 35), pixels(tail @ rotation(self.yaw + theta).T + self.pos))
        pygame.draw.circle(self.screen, (20, 20, 20), pixels(self.pos), 3)
        pygame.draw.circle(self.screen, (210, 45, 60), pixels(self.predator_pos), 5)
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def close(self):
        if self.screen is not None and self.render_mode == "human":
            import pygame
            pygame.display.quit()
        self.screen = self.clock = None
