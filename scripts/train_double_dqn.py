"""
Train Double DQN on ALE/Pacman-v5, continuing from an existing checkpoint.

Standard DQN computes the next-state target using the target network for both
action selection and evaluation:
    target = r + γ * max_a Q_target(s', a)

Double DQN decouples these: the online network picks the action, the target
network scores it. This removes the maximisation bias that causes vanilla DQN
to overestimate Q-values:
    a* = argmax_a Q_online(s', a)
    target = r + γ * Q_target(s', a*)

Everything else (CnnPolicy, replay buffer, callbacks) is identical to the
existing v2-x notebooks, so this can load dqn_v2-8 weights and keep training.

Usage:
    # Continue from v2-8 best model (recommended)
    python train_double_dqn.py --load-from agents/dqn_v2-8/best_model

    # Train from scratch
    python train_double_dqn.py

    # Custom timesteps / W&B run name
    python train_double_dqn.py --load-from agents/dqn_v2-8/best_model \\
        --timesteps 2000000 --run-name double-dqn-v1
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import wandb

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.logger import HParam, TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Double DQN — only train() is overridden
# ---------------------------------------------------------------------------

class DoubleDQN(DQN):
    """
    DQN with the Double DQN target:
        a* = argmax_a Q_online(s', a)
        target = r + γ * Q_target(s', a*)

    Identical to SB3 DQN in every other respect, so existing checkpoints
    (weights, replay buffer, hyperparameters) load without modification.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # --- Double DQN change: two lines replace the vanilla argmax ---
                # Online network selects which action to take
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # Target network evaluates that action's value
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, next_actions)
                # --------------------------------------------------------------
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class HParamCallback(BaseCallback):
    """Logs hyperparameters to TensorBoard at training start."""

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": "DoubleDQN",
            "policy": self.model.policy.__class__.__name__,
            "buffer_size": self.model.buffer_size,
            "batch_size": self.model.batch_size,
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "exploration_fraction": self.model.exploration_fraction,
            "exploration_final_eps": self.model.exploration_final_eps,
            "target_update_interval": self.model.target_update_interval,
            "tau": self.model.tau,
        }
        metric_dict = {
            "eval/mean_reward": 0,
            "eval/mean_ep_length": 0,
            "rollout/ep_rew_mean": 0,
            "train/loss": 0.0,
        }
        from stable_baselines3.common.logger import HParam
        self.logger.record("hparams", HParam(hparam_dict, metric_dict),
                           exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        return True


class WandbCallback(BaseCallback):
    """Logs training scalars to W&B every log_freq steps."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        log = {
            "train/timestep": self.num_timesteps,
            "train/exploration_rate": self.model.exploration_rate,
        }
        if len(self.model.ep_info_buffer) > 0:
            log["train/mean_episode_reward"] = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer]
            )
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

def make_env():
    return Monitor(gym.make(
        "ALE/Pacman-v5",
        render_mode="rgb_array",
        repeat_action_probability=0.25,
        frameskip=4,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", type=str, default=None,
                        help="Path to existing checkpoint to continue from (without .zip). "
                             "Recommended: agents/dqn_v2-8/best_model")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Additional training timesteps (default 1M)")
    parser.add_argument("--run-name", type=str, default="double-dqn-v1")
    parser.add_argument("--project", type=str, default="ale-pacman-v5")
    parser.add_argument("--save-dir", type=str, default="agents/dqn_double")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Hyperparams match dqn_v2-8 for a fair continuation
    custom_objects = {
        "learning_rate": 5e-5,
        "buffer_size": 70_000,
        "batch_size": 64,
        "learning_starts": 100_000,
        "gamma": 0.999,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.005,
        "target_update_interval": 1_000,
        "tensorboard_log": args.save_dir,
        "verbose": 1,
    }

    config = {
        "algorithm": "DoubleDQN",
        "loaded_from": args.load_from or "scratch",
        "total_timesteps": args.timesteps,
        **{k: v for k, v in custom_objects.items() if k not in ("tensorboard_log", "verbose")},
    }

    wandb.init(project=args.project, name=args.run_name, config=config)

    env = make_env()
    eval_env = make_env()

    if args.load_from:
        print(f"\nLoading checkpoint: {args.load_from}.zip")
        model = DoubleDQN.load(
            args.load_from,
            env=env,
            custom_objects=custom_objects,
            seed=args.seed,
        )
    else:
        print("\nTraining Double DQN from scratch")
        model = DoubleDQN(
            policy="CnnPolicy",
            env=env,
            seed=args.seed,
            **{k: v for k, v in custom_objects.items() if k != "verbose"},
            verbose=1,
        )

    eval_callback = WandbEvalCallback(
        eval_env=eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    hparam_callback = HParamCallback()
    wandb_callback = WandbCallback(log_freq=1_000)
    callbacks = CallbackList([hparam_callback, eval_callback, wandb_callback])

    print(f"Algorithm : Double DQN")
    print(f"Device    : {model.device}")
    print(f"Timesteps : {args.timesteps:,}")
    print(f"Save dir  : {args.save_dir}")
    print(f"W&B run   : {wandb.run.url}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name="double_dqn",
        reset_num_timesteps=False,  # keep cumulative timestep count for TensorBoard
        progress_bar=True,
    )

    save_path = os.path.join(args.save_dir, "ALE-Pacman-v5")
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    wandb.finish()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
