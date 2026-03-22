"""Multi-worker PPO training loop with checkpoint resumption.

This script orchestrates distributed reinforcement-learning training for
the fish escape task.  Each worker launches its own ANSYS Fluent instance,
trains a PPO agent, and shares best-model information through a
:class:`SharedTrainingManager` backed by ``multiprocessing`` primitives.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
import traceback
from typing import Any, Dict, List, Optional

import multiprocessing as mp

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from EnvFluent import FluentEnv


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


class EnhancedCallback(BaseCallback):
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
            "./saved_models", "saved_model.zip"
        )
        self.global_saved_vecnorm = os.path.join(
            "./saved_models", "saved_vecnormalize.pkl"
        )
        os.makedirs("./saved_models", exist_ok=True)

    def _save_checkpoint(
        self, model_path: str, vecnorm_path: str
    ) -> None:
        """Save the model and ``VecNormalize`` statistics to disk."""
        self.model.save(model_path)
        try:
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(vecnorm_path)
        except Exception as e:
            print(
                f"[Rank {self.rank}] Warning: "
                f"failed to save VecNormalize: {e}"
            )

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
                        "obstacle_distance", "target_distance", "reward",
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
                "reward": float(reward),
                "success": info.get("success", False),
                "failed": info.get("failed", False),
                "failure_reason": info.get("failure_reason", ""),
            })

            self.current_episode_reward += reward

            if done:
                self._handle_episode_end(info)

        except Exception as e:
            print(f"Error in callback for rank {self.rank}: {e}")
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
        self.model.save(model_path)

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


def build_env_with_optional_resume(
    rank: int,
    log_path: str,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    local_saved_vecnorm: Optional[str] = None,
    global_saved_vecnorm: Optional[str] = None,
) -> VecNormalize:
    """Create and optionally restore a ``VecNormalize``-wrapped environment.

    Attempts to reload statistics in priority order: local checkpoint,
    global checkpoint, then fresh wrapper.
    """

    def make_env():
        def _init():
            env = FluentEnv(max_steps=800, simu_name=f"CFD_{rank}")
            return Monitor(env, os.path.join(log_path, "monitor"))
        return _init

    base_env = DummyVecEnv([make_env()])

    if local_saved_vecnorm and os.path.exists(local_saved_vecnorm):
        print(f"Worker {rank}: Loading VecNormalize from {local_saved_vecnorm}")
        env = VecNormalize.load(local_saved_vecnorm, base_env)
        env.training = True
    elif global_saved_vecnorm and os.path.exists(global_saved_vecnorm):
        print(
            f"Worker {rank}: Loading GLOBAL VecNormalize "
            f"from {global_saved_vecnorm}"
        )
        env = VecNormalize.load(global_saved_vecnorm, base_env)
        env.training = True
    else:
        env = VecNormalize(
            base_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
        )

    return env


def train_with_rank(
    rank: int,
    num_workers: int,
    manager: SharedTrainingManager,
    total_timesteps: int,
    use_lstm: bool = False,
) -> None:
    """Train a PPO agent on a single Fluent instance.

    Handles staggered startup, checkpoint resumption, and graceful shutdown.
    """
    env: Optional[VecNormalize] = None
    model: Optional[PPO] = None
    save_path = os.path.join("./saved_models", f"worker_{rank}")

    try:
        # Stagger startup to avoid resource contention
        startup_delay = rank * 30
        print(f"Worker {rank}: Waiting {startup_delay}s before launch...")
        time.sleep(startup_delay)

        manager.update_worker_status(rank, SharedTrainingManager.STATUS_RUNNING)

        # Seed for reproducibility
        torch.manual_seed(rank * 42)
        np.random.seed(rank * 42)

        # Directory setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_path = os.path.join("./logs", f"worker_{rank}")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # Checkpoint paths
        local_saved_model = os.path.join(save_path, "saved_model.zip")
        local_saved_vecnorm = os.path.join(save_path, "saved_vecnormalize.pkl")
        global_saved_model = os.path.join("./saved_models", "saved_model.zip")
        global_saved_vecnorm = os.path.join(
            "./saved_models", "saved_vecnormalize.pkl"
        )

        # Build environment
        env = build_env_with_optional_resume(
            rank,
            log_path,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            local_saved_vecnorm=local_saved_vecnorm,
            global_saved_vecnorm=global_saved_vecnorm,
        )

        # Network architecture
        if use_lstm:
            policy_kwargs: Dict[str, Any] = dict(
                net_arch=dict(pi=[1024, 512, 256], vf=[512, 256, 128]),
                lstm_hidden_size=256,
                enable_critic_lstm=True,
                lstm_layers=2,
            )
            policy_type = "MlpLstmPolicy"
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 512, 256], vf=[512, 256, 128]),
                activation_fn=torch.nn.ReLU,
            )
            policy_type = "MlpPolicy"

        # Resume from checkpoint or create a fresh model
        if os.path.exists(local_saved_model):
            print(f"Worker {rank}: Resuming from {local_saved_model}")
            model = PPO.load(local_saved_model, env=env, device=device)
        elif os.path.exists(global_saved_model):
            print(f"Worker {rank}: Resuming from GLOBAL {global_saved_model}")
            model = PPO.load(global_saved_model, env=env, device=device)
        else:
            model = PPO(
                policy_type,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=32,
                batch_size=16,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.98,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                tensorboard_log=log_path,
                device=device,
            )

        # Create callback and start training
        callback = EnhancedCallback(save_path, rank, manager, verbose=1)

        print(f"Worker {rank}: Starting training for {total_timesteps} steps")
        print(
            f"Worker {rank}: Observation space: {env.observation_space}"
        )
        print(f"Worker {rank}: Action space: {env.action_space}")

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

        # Save final model and normalisation statistics
        final_path = os.path.join(save_path, "final_model.zip")
        model.save(final_path)
        try:
            env.save(os.path.join(save_path, "vec_normalize.pkl"))
        except Exception as e:
            print(f"Worker {rank}: Warning saving vec_normalize.pkl: {e}")

        manager.update_worker_status(rank, SharedTrainingManager.STATUS_DONE)
        print(f"Worker {rank}: Training completed successfully")

    except KeyboardInterrupt:
        print(f"Worker {rank}: Interrupted. Saving backup...")
        try:
            backup_path = os.path.join(save_path, "interrupted_model.zip")
            if model is not None:
                model.save(backup_path)
        except Exception:
            pass
        manager.update_worker_status(rank, SharedTrainingManager.STATUS_ERROR)

    except Exception as e:
        print(f"Worker {rank}: Error: {e}")
        traceback.print_exc()
        manager.update_worker_status(rank, SharedTrainingManager.STATUS_ERROR)

    finally:
        if env is not None:
            try:
                env.close()
                print(f"Worker {rank}: Environment closed")
            except Exception as e:
                print(f"Worker {rank}: Error closing environment: {e}")


def monitor_workers(
    manager: SharedTrainingManager,
    num_workers: int,
    check_interval: int = 60,
) -> None:
    """Periodically report worker status until all workers finish."""
    status_labels = ["Not started", "Running", "Error", "Done"]

    while True:
        time.sleep(check_interval)

        status_counts = [0, 0, 0, 0]
        for i in range(num_workers):
            status_counts[manager.worker_status[i]] += 1

        print("\n=== Worker Status Monitor ===")
        for label, count in zip(status_labels, status_counts):
            print(f"  {label}: {count}")
        print(f"  Global best reward: {manager.get_best_reward():.2f}")

        # Exit when no workers are running
        if status_counts[SharedTrainingManager.STATUS_RUNNING] == 0:
            print("All workers have stopped.")
            break


def main() -> None:
    """Launch multi-worker PPO training with checkpoint resumption."""
    # Set multiprocessing start method
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)
    else:
        mp.set_start_method("fork", force=True)

    # Training hyper-parameters
    num_workers = 1
    total_timesteps = 20_000
    use_lstm = False

    print("=" * 60)
    print("FISH ESCAPE TRAINING (stable + checkpoint resumption)")
    print("=" * 60)
    print(f"  Workers            : {num_workers}")
    print(f"  Timesteps / worker : {total_timesteps}")
    print(f"  LSTM policy        : {use_lstm}")
    print(f"  Task               : Navigate fish to target while escaping")
    print(f"  Startup strategy   : Staggered (30 s between workers)")
    print("=" * 60)

    manager = SharedTrainingManager(num_workers)

    # Start monitor process
    monitor_process = mp.Process(
        target=monitor_workers, args=(manager, num_workers)
    )
    monitor_process.start()

    # Start training processes
    processes: List[mp.Process] = []
    for rank in range(num_workers):
        p = mp.Process(
            target=train_with_rank,
            args=(rank, num_workers, manager, total_timesteps, use_lstm),
        )
        p.start()
        processes.append(p)
        print(f"Worker {rank} process launched")

    try:
        for p in processes:
            p.join()
        monitor_process.terminate()
        monitor_process.join()

    except KeyboardInterrupt:
        print("\nInterrupted! Terminating processes...")
        for p in processes:
            p.terminate()
        monitor_process.terminate()
        for p in processes:
            p.join()
        monitor_process.join()

    # Report best model
    best_model_path = manager.get_best_model_path()
    best_reward = manager.get_best_reward()
    print(
        f"\nTraining completed.  Best model: {best_model_path} "
        f"(reward: {best_reward:.2f})"
    )

    # Copy best model to a well-known location
    if best_model_path and os.path.exists(best_model_path):
        final_best_path = "./saved_models/best_obstacle_avoidance_model.zip"
        try:
            shutil.copy(best_model_path, final_best_path)
            print(f"Best model copied to {final_best_path}")
        except Exception as e:
            print(f"Failed to copy best model: {e}")
    else:
        print("No valid best model found.")


if __name__ == "__main__":
    main()
