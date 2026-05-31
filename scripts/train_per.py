"""
Train DQN with Prioritized Experience Replay (PER) on ALE/Pacman-v5.
Uses the same hyperparameters as dqn_v2-8 for a fair comparison.

PER is implemented via a custom ReplayBuffer subclass — no extra dependencies
beyond what's already in the repo (SB3 + numpy).

W&B is used for logging. The baseline DQN history can be uploaded separately
via upload_history_to_wandb.py and compared in the same W&B project.

Usage:
    python train_per.py
    python train_per.py --timesteps 500000 --run-name my-per-run
    python train_per.py --no-per  # train vanilla DQN for a direct local comparison

M2 Mac note: trains on CPU (MPS not supported by SB3). 500k steps takes ~2-4 hours.
"""

import argparse
import os
import numpy as np
import torch
import wandb
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples

gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# Prioritized Experience Replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional PER (Schaul et al., 2015).
    Adds priority-weighted sampling on top of SB3's standard ReplayBuffer.
    """

    def __init__(self, *args, alpha: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._priorities = np.ones(self.buffer_size, dtype=np.float32)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos  # position that will be written
        super().add(*args, **kwargs)
        self._priorities[idx] = self._max_priority

    def sample(self, batch_size: int, beta: float = 0.4, env=None) -> ReplayBufferSamples:
        n = self.buffer_size if self.full else self.pos
        priorities = self._priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Store for update step
        self._last_indices = indices
        self._last_weights = weights.astype(np.float32)

        return self._get_samples(indices, env=env)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + 1e-6
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max())


# ---------------------------------------------------------------------------
# W&B + beta annealing callback
# ---------------------------------------------------------------------------

class WandbPERCallback(BaseCallback):
    """Logs training metrics to W&B and anneals PER beta from beta_start → 1.0."""

    def __init__(self, total_timesteps: int, beta_start: float = 0.4,
                 use_per: bool = True, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.use_per = use_per
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        progress = self.num_timesteps / self.total_timesteps
        log = {
            "train/timestep": self.num_timesteps,
            "train/exploration_rate": self.model.exploration_rate,
        }

        if self.use_per and hasattr(self.model.replay_buffer, "_last_weights"):
            beta = self.beta_start + progress * (1.0 - self.beta_start)
            log["per/beta"] = beta
            log["per/max_priority"] = float(self.model.replay_buffer._max_priority)
            log["per/mean_weight"] = float(self.model.replay_buffer._last_weights.mean())

        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            log["train/mean_episode_reward"] = np.mean(ep_rewards)

        wandb.log(log, step=self.num_timesteps)
        return True


class WandbEvalCallback(EvalCallback):
    """EvalCallback that also logs eval results to W&B."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                wandb.log({
                    "eval/mean_reward": self.last_mean_reward,
                    "eval/best_mean_reward": self.best_mean_reward,
                }, step=self.num_timesteps)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_env(render_mode="rgb_array"):
    env = gym.make(
        "ALE/Pacman-v5",
        render_mode=render_mode,
        repeat_action_probability=0.25,
        frameskip=4,
    )
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default 500k for local runs)")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project", type=str, default="ale-pacman-v5")
    parser.add_argument("--save-dir", type=str, default="agents/dqn_per")
    parser.add_argument("--no-per", action="store_true",
                        help="Disable PER and train vanilla DQN (for local comparison)")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER alpha (priority exponent)")
    parser.add_argument("--beta-start", type=float, default=0.4, help="PER beta start (IS weight exponent)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_per = not args.no_per
    run_name = args.run_name or ("dqn-per" if use_per else "dqn-baseline")
    save_dir = args.save_dir if use_per else args.save_dir.replace("per", "baseline")
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "algorithm": "DQN+PER" if use_per else "DQN",
        "environment": "ALE/Pacman-v5",
        "total_timesteps": args.timesteps,
        "per_alpha": args.alpha if use_per else None,
        "per_beta_start": args.beta_start if use_per else None,
        # Match dqn_v2-8 hyperparameters
        "learning_rate": 5e-5,
        "buffer_size": 70_000,
        "batch_size": 64,
        "learning_starts": 50_000,   # reduced from 100k to suit shorter local runs
        "gamma": 0.999,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.005,
        "target_update_interval": 1000,
        "seed": args.seed,
    }

    wandb.init(project=args.project, name=run_name, config=config)

    env = make_env()
    eval_env = make_env()

    replay_buffer_class = PrioritizedReplayBuffer if use_per else ReplayBuffer
    replay_buffer_kwargs = {"alpha": args.alpha} if use_per else {}

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        learning_starts=config["learning_starts"],
        gamma=config["gamma"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        seed=args.seed,
        device="cpu",   # MPS not supported by SB3; CPU is stable on M2
    )

    eval_callback = WandbEvalCallback(
        eval_env=eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    wandb_callback = WandbPERCallback(
        total_timesteps=args.timesteps,
        beta_start=args.beta_start,
        use_per=use_per,
        log_freq=1_000,
    )

    print(f"\nTraining {'DQN+PER' if use_per else 'DQN'} for {args.timesteps:,} steps")
    print(f"Device: {model.device} | Saving to: {save_dir}")
    print(f"W&B run: {wandb.run.url}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, wandb_callback],
        progress_bar=True,
    )

    model.save(os.path.join(save_dir, "ALE-Pacman-v5"))
    print(f"\nModel saved to {save_dir}/ALE-Pacman-v5.zip")

    wandb.finish()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
