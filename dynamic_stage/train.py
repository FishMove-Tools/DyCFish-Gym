"""PPO training script for the fish escape environment.

This script trains a PPO agent (from Stable-Baselines3) on the
simplified :class:`FishEnv` dynamics environment with real-time Pygame
rendering, periodic checkpointing, and per-episode reward logging.
"""

from __future__ import annotations

import os

# Disable tqdm rich backend to avoid errors on exit.
os.environ["TQDM_DISABLE_RICH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from typing import List, Optional

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    VecNormalize,
)

from fish_env import FishEnv  # noqa: E402

# Directory setup
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

REWARD_LOG_FILE = "./logs/episode_rewards.txt"
if not os.path.exists(REWARD_LOG_FILE):
    with open(REWARD_LOG_FILE, "w") as f:
        pass


# Environment factory


def make_env() -> Monitor:
    """Create a :class:`FishEnv` wrapped with :class:`Monitor`."""
    env = FishEnv(
        render_mode="human",
        max_steps=2000,
        predator_speed=0.3,
        render_skip=10,
        enable_step_logging=False,
    )
    env = Monitor(env, "./logs")
    return env


env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)


# Callbacks
class RealTimeRenderCallback(BaseCallback):
    """Trigger environment rendering at a configurable call interval.

    The actual rendering frequency is governed by ``render_skip`` inside
    :class:`FishEnv`.
    """

    def __init__(self, every_calls: int = 1, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.every_calls = max(1, int(every_calls))
        self._calls: int = 0

    def _on_step(self) -> bool:
        self._calls += 1
        if (self._calls % self.every_calls) == 0:
            try:
                # Unwrap to the innermost env (past VecNormalize / Monitor)
                env0 = self.training_env
                if hasattr(env0, "envs"):
                    env0 = env0.envs[0]
                if hasattr(env0, "env"):
                    env0 = env0.env
                if getattr(env0, "render_mode", None) == "human":
                    env0.render()
            except Exception:
                pass
        return True


class RewardLoggingCallback(BaseCallback):
    """Log per-episode reward and step count; append a summary at training end."""

    def __init__(self, log_file: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_file = log_file

        self.episode_rewards: List[float] = []
        self.current_episode_reward: float = 0.0
        self.episode_count: int = 0
        self.total_steps: int = 0
        self.episode_step_count: int = 0

        # Write log header
        with open(self.log_file, "w") as f:
            f.write("Training Progress:\n")
            f.write("Episode,Episode Steps,Episode Reward,Total Steps\n")

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.current_episode_reward += reward
        self.total_steps += 1
        self.episode_step_count += 1

        if done:
            self.episode_count += 1
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_count},"
                    f"{self.episode_step_count},"
                    f"{self.current_episode_reward:.2f},"
                    f"{self.total_steps}\n"
                )
            print(
                f"Episode {self.episode_count} | "
                f"steps: {self.episode_step_count} | "
                f"reward: {self.current_episode_reward:.2f}"
            )
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.episode_step_count = 0

        return True

    def _on_training_end(self) -> None:
        """Append a training summary to the log file."""
        if not self.episode_rewards:
            return
        with open(self.log_file, "a") as f:
            f.write("\nTraining Summary:\n")
            f.write(f"Total Steps: {self.total_steps}\n")
            f.write(f"Total Episodes: {self.episode_count}\n")
            f.write(f"Average Reward: {np.mean(self.episode_rewards):.2f}\n")
            f.write(f"Max Reward: {np.max(self.episode_rewards):.2f}\n")
            f.write(f"Min Reward: {np.min(self.episode_rewards):.2f}\n")


# Model & training

checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path="./models/", name_prefix="fish_model",
)
render_callback = RealTimeRenderCallback(every_calls=1)
reward_logging_callback = RewardLoggingCallback(log_file=REWARD_LOG_FILE)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.005,
    clip_range=0.2,
    tensorboard_log="./tensorboard/",
)

try:
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, render_callback, reward_logging_callback],
        progress_bar=False,
    )
    model.save("./models/fish_final")
    print(f"Training done. Episode rewards logged at {REWARD_LOG_FILE}")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    model.save("./models/fish_interrupted")

finally:
    env.close()
