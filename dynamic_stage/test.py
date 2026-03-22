"""Evaluation and video recording script for the fish escape agent.

This script loads a trained PPO model, replays it in the :class:`FishEnv`
environment for a configurable number of episodes, and records a video
of the agent's behaviour using OpenCV.
"""

from __future__ import annotations

import os

# Disable tqdm rich backend to avoid errors on exit.
os.environ["TQDM_DISABLE_RICH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from typing import Optional, Union

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    VecNormalize,
)

from fish_env import FishEnv  # noqa: E402

# Configuration

MODEL_PATH = "./models/fish_model_800000_steps.zip"
VECNORM_PATH = "./models/vecnormalize.pkl"
VIDEO_PATH = "./logs/test_escape.mp4"
EPISODES = 5


# Environment factory

def make_env_for_eval(render_mode: str = "rgb_array") -> Monitor:
    """Create a :class:`FishEnv` for evaluation with training-consistent parameters."""
    env = FishEnv(
        render_mode=render_mode, max_steps=2000, predator_speed=0.3,
    )
    env = Monitor(env, "./logs")
    return env


# Main evaluation loop

def main() -> None:
    """Load a trained model, evaluate it, and record a video."""
    os.makedirs("./logs", exist_ok=True)

    # 1) Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device="cpu")

    # 2) Build evaluation environment, restoring VecNormalize if available
    env: Union[VecNormalize, FishEnv, Monitor]
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize stats from: {VECNORM_PATH}")
        venv = DummyVecEnv(
            [lambda: make_env_for_eval(render_mode="rgb_array")]
        )
        env = VecNormalize.load(VECNORM_PATH, venv)
        env.training = False
        env.norm_reward = False
    else:
        print(
            "VecNormalize stats not found; evaluating with un-normalised "
            "observations (still functional)."
        )
        env = make_env_for_eval(render_mode="rgb_array")

    # 3) Initialise video writer (need a first frame for dimensions)
    if isinstance(env, VecNormalize):
        obs = env.reset()
        # Step once to ensure FishEnv internal attributes (A, w) exist
        action = np.zeros(
            (1, env.action_space.shape[0]), dtype=np.float32,
        )
        _obs, _reward, _done, _info = env.step(action)
        frame = env.venv.envs[0].render()
        height, width = frame.shape[:2]
    else:
        obs, _ = env.reset()
        if hasattr(env, "action_space"):
            zero_act = env.action_space.sample() * 0.0
            _obs, _reward, _done, _trunc, _info = env.step(zero_act)
        frame = env.render()
        height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(VIDEO_PATH, fourcc, 30, (width, height))
    print(f"Video writer opened: {VIDEO_PATH} (size: {width}×{height})")

    # 4) Episode loop with recording
    for ep in range(1, EPISODES + 1):
        if isinstance(env, VecNormalize):
            obs = env.reset()
        else:
            obs, _ = env.reset()

        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            if isinstance(env, VecNormalize):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                frame = env.venv.envs[0].render()
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                frame = env.render()

            ep_reward += float(reward)
            step += 1

            if frame is not None:
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"[Episode {ep}] steps={step}, reward={ep_reward:.2f}")

    vw.release()
    if hasattr(env, "close"):
        env.close()
    print(f"Evaluation complete. Video saved to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
