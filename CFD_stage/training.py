"""Multi-worker PPO training with explicit Fluent setup.

This script orchestrates distributed reinforcement-learning training for
the explicitly configured CFD target task.  Each worker launches its own ANSYS Fluent instance,
trains a PPO agent, and shares best-model information through a
:class:`SharedTrainingManager` backed by ``multiprocessing`` primitives.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path
import sys
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import multiprocessing as mp

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from CFD_stage.interface import interface_contract


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return parsed


def resolve_initializer(spec):
    """Resolve the user-supplied complete Fluent setup/reset callback."""
    if ":" not in spec:
        raise ValueError("--initializer must have the form module:function.")
    module_name, function_name = spec.rsplit(":", 1)
    if not module_name or not function_name:
        raise ValueError("--initializer must have the form module:function.")
    callback = getattr(importlib.import_module(module_name), function_name)
    if not callable(callback):
        raise ValueError("The specified initializer is not callable.")
    return callback


def cfd_contract(target, initializer):
    return {**interface_contract(),
            "target_position": list(target), "initializer": initializer}








class SharedTrainingManager:
    """Thread-safe manager for coordinating multiple training workers.

    Maintains a shared best-reward value, best-model path, and worker
    status array via ``multiprocessing`` synchronisation primitives.
    """

    # Worker status codes
    STATUS_NOT_STARTED = 0
    STATUS_RUNNING = 1
    STATUS_ERROR = 2
    STATUS_DONE = 3

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.reward_queue: mp.Queue = mp.Queue()
        self.model_queue: mp.Queue = mp.Queue()
        self.best_reward = mp.Value("d", -float("inf"))
        self.best_model_path = mp.Array("c", b"\x00" * 500)
        self.lock = mp.Lock()
        self.worker_status = mp.Array("i", [0] * num_workers)

    def update_worker_status(self, rank: int, status: int) -> None:
        """Set the status code for a given worker."""
        self.worker_status[rank] = status

    def update_best_model(
        self, rank: int, reward: float, model_path: str
    ) -> bool:
        """Atomically update the global best model if *reward* improves.

        Returns ``True`` if the global best was updated.
        """
        with self.lock:
            if reward > self.best_reward.value:
                self.best_reward.value = reward
                path_bytes = model_path.encode("utf-8")[:499]
                self.best_model_path.value = path_bytes + b"\x00" * (
                    500 - len(path_bytes)
                )
                return True
        return False

    def get_best_model_path(self) -> str:
        """Return the path to the current global best model."""
        with self.lock:
            return self.best_model_path.value.decode("utf-8").rstrip("\x00")

    def get_best_reward(self) -> float:
        """Return the current global best mean reward."""
        with self.lock:
            return self.best_reward.value


class CallbackLogic:
    """PPO callback that logs per-episode statistics and manages checkpoints.

    At the end of every episode the callback saves snapshots, updates
    local/global best checkpoints, writes CSV logs, and monitors
    consecutive Fluent failures.
    """

    def __init__(
        self,
        save_path: str,
        rank: int,
        manager: SharedTrainingManager,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.should_stop = False
        self.save_path = save_path
        self.rank = rank
        self.manager = manager

        # Episode counters
        self.episode_count: int = 0
        self.episode_rewards: List[float] = []
        self.current_episode_reward: float = 0.0
        self.best_mean_reward: float = -np.inf

        # Consecutive-failure monitoring
        self.consecutive_failures: int = 0
        self.max_consecutive_failures: int = 5

        # Log file paths
        self.reward_log_path = os.path.join(
            save_path, f"rewards_rank{rank}.csv"
        )
        self.action_log_folder = os.path.join(
            save_path, f"actions_rank{rank}"
        )
        self.performance_log_path = os.path.join(
            save_path, f"performance_rank{rank}.csv"
        )
        os.makedirs(self.action_log_folder, exist_ok=True)

        # Initialise CSV log headers
        with open(self.reward_log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "Episode", "Reward", "Mean_Reward", "Best_Global_Reward",
                "Success", "Failure_Reason", "Episode_Length",
            ])

        with open(self.performance_log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "Episode", "Final_Target_Distance", "Min_Obstacle_Distance",
                "Simulation_Time", "Success_Rate", "Avg_Turning_Action",
                "Avg_Period_Action", "Consecutive_Failures",
            ])

        self.step_counter: int = 0
        self.episode_file: Optional[Any] = None
        self.episode_writer: Optional[csv.DictWriter] = None

        # Per-episode performance statistics
        self.episode_turning_actions: List[float] = []
        self.episode_period_actions: List[float] = []
        self.min_obstacle_distance: float = float("inf")
        self.success_count: int = 0

        # Fixed checkpoint paths (for resumption)
        self.local_saved_model = os.path.join(self.save_path, "saved_model.zip")
        self.local_saved_vecnorm = os.path.join(
            self.save_path, "saved_vecnormalize.pkl"
        )
        self.global_saved_model = os.path.join(
            str(Path(self.save_path).parent), "saved_model.zip"
        )
        self.global_saved_vecnorm = os.path.join(
            str(Path(self.save_path).parent), "saved_vecnormalize.pkl"
        )
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, model_path, vecnorm_path):
        """Save the model and its current normalization statistics."""
        self.model.save(model_path)
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(vecnorm_path)

    def _on_training_end(self):
        if self.episode_file is not None:
            self.episode_file.close()
            self.episode_file = None

    def _on_step(self) -> bool:  # noqa: C901
        """Called after every environment step."""
        try:
            reward = self.locals.get("rewards", [0])[0]
            done = self.locals.get("dones", [False])[0]

            # Open a new per-episode action log if needed
            if self.episode_file is None:
                episode_filename = os.path.join(
                    self.action_log_folder,
                    f"episode_{self.episode_count + 1}_actions.csv",
                )
                self.episode_file = open(
                    episode_filename, "w", newline="", buffering=1
                )
                self.episode_writer = csv.DictWriter(
                    self.episode_file,
                    fieldnames=[
                        "step", "simulation_time", "fish_x", "fish_y",
                        "fish_theta", "turning_action", "period_action",
                        "obstacle_distance", "target_distance", "normalized_reward",
                        "success", "failed", "failure_reason",
                    ],
                )
                self.episode_writer.writeheader()

                # Reset per-episode statistics
                self.episode_turning_actions = []
                self.episode_period_actions = []
                self.min_obstacle_distance = float("inf")

            # Gather step info
            info: Dict[str, Any] = self.locals.get("infos", [{}])[0]
            actions = self.locals.get("actions", [0])
            action = actions[0] if len(actions) > 0 else 0

            turning_action = info.get("turning_action", 0)
            period_action = info.get("period_action", 1)
            obstacle_distance = info.get("obstacle_distance", float("inf"))

            self.episode_turning_actions.append(turning_action)
            self.episode_period_actions.append(period_action)
            self.min_obstacle_distance = min(
                self.min_obstacle_distance, obstacle_distance
            )

            self.step_counter += 1
            self.episode_writer.writerow({
                "step": self.step_counter,
                "simulation_time": info.get("simulation_time", 0),
                "fish_x": info.get("fish_position", [0, 0])[0],
                "fish_y": info.get("fish_position", [0, 0])[1],
                "fish_theta": info.get("fish_orientation", 0),
                "turning_action": turning_action,
                "period_action": period_action,
                "obstacle_distance": obstacle_distance,
                "target_distance": info.get("target_distance", float("inf")),
                "normalized_reward": float(reward),
                "success": info.get("success", False),
                "failed": info.get("failed", False),
                "failure_reason": info.get("failure_reason", ""),
            })

            self.current_episode_reward += float(reward)
            if done:
                self._handle_episode_end(info)
                if self.should_stop:
                    return False

        except Exception as e:
            print(f"Error in callback for rank {self.rank}: {e}")
            self.should_stop = True
            self.manager.update_worker_status(self.rank, SharedTrainingManager.STATUS_ERROR)
            return False

        return True

    def _handle_episode_end(self, info: Dict[str, Any]) -> None:
        """Process logging, checkpointing, and failure tracking."""
        self.episode_count += 1
        self.episode_rewards.append(self.current_episode_reward)

        # Track consecutive Fluent failures
        failure_reason = info.get("failure_reason", "")
        fluent_failures = {
            "fluent_connection_lost",
            "fluent_exception",
            "fluent_step_exception",
        }
        if failure_reason in fluent_failures:
            self.consecutive_failures += 1
            print(
                f"[Rank {self.rank}] Fluent connection failure "
                f"({self.consecutive_failures}/{self.max_consecutive_failures})"
            )
        else:
            self.consecutive_failures = 0

        if self.consecutive_failures >= self.max_consecutive_failures:
            self.should_stop = True
            print(
                f"[Rank {self.rank}] Too many consecutive failures; "
                f"marking worker as errored."
            )
            self.manager.update_worker_status(
                self.rank, SharedTrainingManager.STATUS_ERROR
            )

        # Update success statistics
        if info.get("success", False):
            self.success_count += 1

        # Compute rolling mean reward
        window = self.episode_rewards[-10:]
        mean_reward = float(np.mean(window))

        # Success rate
        success_rate = self.success_count / self.episode_count

        # Save per-episode snapshot
        model_path = os.path.join(
            self.save_path,
            f"model_rank{self.rank}_ep{self.episode_count}.zip",
        )
        self._save_checkpoint(model_path, str(Path(model_path).with_suffix(".vecnormalize.pkl")))

        # Update local / global best checkpoints
        best_status = ""
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self._save_checkpoint(
                self.local_saved_model, self.local_saved_vecnorm
            )
            is_global_best = self.manager.update_best_model(
                self.rank, mean_reward, model_path
            )
            if is_global_best:
                best_status = "GLOBAL BEST!"
                try:
                    self._save_checkpoint(
                        self.global_saved_model, self.global_saved_vecnorm
                    )
                except Exception as e:
                    print(
                        f"[Rank {self.rank}] Warning: "
                        f"failed to update global saved_model: {e}"
                    )
            else:
                best_status = "Local Best"

        global_best = self.manager.get_best_reward()

        # Append to reward CSV
        with open(self.reward_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count,
                self.current_episode_reward,
                mean_reward,
                global_best,
                info.get("success", False),
                info.get("failure_reason", ""),
                self.step_counter,
            ])

        # Append to performance CSV
        avg_turning = (
            float(np.mean(self.episode_turning_actions))
            if self.episode_turning_actions
            else 0.0
        )
        avg_period = (
            float(np.mean(self.episode_period_actions))
            if self.episode_period_actions
            else 1.0
        )
        with open(self.performance_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count,
                info.get("target_distance", float("inf")),
                self.min_obstacle_distance,
                info.get("simulation_time", 0),
                success_rate,
                avg_turning,
                avg_period,
                self.consecutive_failures,
            ])

        print(
            f"[Rank {self.rank:02d}|Ep {self.episode_count:03d}] "
            f"Reward: {self.current_episode_reward:.2f} | "
            f"Mean10: {mean_reward:.2f} | "
            f"Success Rate: {success_rate:.2%} | "
            f"Target Dist: {info.get('target_distance', float('inf')):.2f} | "
            f"Global Best: {global_best:.2f} {best_status}"
        )

        # Reset per-episode state
        if self.episode_file:
            self.episode_file.close()
        self.episode_file = None
        self.episode_writer = None
        self.step_counter = 0
        self.current_episode_reward = 0.0


def create_callback(*args, **kwargs):
    # Keep import/CLI inspection safe without importing or starting PPO.
    from stable_baselines3.common.callbacks import BaseCallback

    class EnhancedCallback(CallbackLogic, BaseCallback):
        pass

    return EnhancedCallback(*args, **kwargs)


def build_env(rank, log_path, env_kwargs):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from CFD_stage.EnvFluent import FluentEnv

    def make_env():
        return Monitor(FluentEnv(max_steps=800, simu_name=f"CFD_{rank}", **env_kwargs),
                       str(Path(log_path) / "monitor"))

    return VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True, clip_obs=10.0)


def train_with_rank(rank, manager, settings):
    """Train the explicitly configured CFD target task."""
    import torch
    from stable_baselines3 import PPO

    env, model, callback = None, None, None
    output_dir = Path(settings["output_dir"])
    save_path = output_dir / "saved_models" / f"worker_{rank}"
    log_path = output_dir / "logs" / f"worker_{rank}"
    seed = settings["seed"] + rank * 42
    try:
        manager.update_worker_status(rank, SharedTrainingManager.STATUS_RUNNING)
        time.sleep(rank * 30)
        for folder in (save_path, log_path):
            folder.mkdir(parents=True, exist_ok=True)
        initializer = resolve_initializer(settings["initializer"])
        env = build_env(rank, log_path,
            {"target_position": tuple(settings["target"]), "reward_function": "target",
             "initializer": initializer, "work_dir": output_dir / "runs" / f"CFD_{rank}"})
        env.seed(seed)
        model = PPO("MlpPolicy", env,
            policy_kwargs=dict(net_arch=dict(pi=[1024, 512, 256], vf=[512, 256, 128]),
                               activation_fn=torch.nn.ReLU),
            learning_rate=3e-4, n_steps=32, batch_size=16, n_epochs=10,
            gamma=0.995, gae_lambda=0.98, clip_range=0.2, ent_coef=0.005,
            vf_coef=0.5, max_grad_norm=0.5, verbose=0,
            tensorboard_log=str(log_path),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), seed=seed)
        callback = create_callback(str(save_path), rank, manager)
        model.learn(total_timesteps=settings["timesteps"], callback=callback)
        model.save(str(save_path / "final_model.zip"))
        env.save(str(save_path / "final_model.vecnormalize.pkl"))
        if callback.should_stop:
            manager.update_worker_status(rank, SharedTrainingManager.STATUS_ERROR)
            print(f"Worker {rank}: stopped after repeated solver failures.")
        else:
            manager.update_worker_status(rank, SharedTrainingManager.STATUS_DONE)
            print(f"Worker {rank}: training completed.")
    except KeyboardInterrupt:
        if model is not None and env is not None:
            model.save(str(save_path / "interrupted_model.zip"))
            env.save(str(save_path / "interrupted_model.vecnormalize.pkl"))
        manager.update_worker_status(rank, SharedTrainingManager.STATUS_ERROR)
    except Exception as error:
        print(f"Worker {rank}: {error}")
        traceback.print_exc()
        manager.update_worker_status(rank, SharedTrainingManager.STATUS_ERROR)
    finally:
        if callback is not None:
            callback._on_training_end()
        if env is not None:
            env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="CFD target-task training; not a Table 4 reproduction or automatic ROM transfer.")
    parser.add_argument("--target", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--initializer", required=True, metavar="MODULE:FUNCTION",
        help="Complete case/UDF setup and reset callback (solver, work_dir) returning the real initial 7D state.")
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--timesteps", type=positive_int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) for value in args.target):
        parser.error("--target must contain two finite coordinates.")
    # Reject missing setup before any worker/solver is launched.
    try:
        resolve_initializer(args.initializer)
    except (ValueError, OSError, ImportError, AttributeError) as error:
        parser.error(str(error))
    # Dependencies are also checked in the parent process, without creating Fluent.
    import torch  # noqa: F401
    from stable_baselines3 import PPO  # noqa: F401
    from CFD_stage.EnvFluent import FluentEnv  # noqa: F401

    mp.set_start_method("spawn", force=True)
    settings = {"output_dir": str(args.output_dir.resolve()), "target": args.target,
                "initializer": args.initializer, "timesteps": args.timesteps,
                "seed": args.seed}
    manager = SharedTrainingManager(args.workers)
    processes = []
    try:
        for rank in range(args.workers):
            process = mp.Process(target=train_with_rank, args=(rank, manager, settings))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        raise
    failed = any(process.exitcode != 0 for process in processes) or any(
        value != SharedTrainingManager.STATUS_DONE for value in manager.worker_status)
    print(f"Best CFD training checkpoint: {manager.get_best_model_path() or 'none'}")
    if failed:
        raise SystemExit("One or more CFD workers failed; see worker logs.")


if __name__ == "__main__":
    main()
