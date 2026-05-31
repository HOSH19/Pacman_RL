import argparse
import os
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork

gym.register_envs(ale_py)


class DuelingQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        features_dim = self.features_dim
        n_actions    = self.action_space.n
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(features_dim, 64), nn.ReLU(), nn.Linear(64, n_actions),
        )
        del self.q_net

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features  = self.extract_features(obs, self.features_extractor)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingCnnPolicy(CnnPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._priorities = np.ones(self.buffer_size, dtype=np.float32)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self._priorities[idx] = self._max_priority

    def sample(self, batch_size: int, beta: float = 0.4, env=None) -> ReplayBufferSamples:
        n = self.buffer_size if self.full else self.pos
        priorities = self._priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(n, size=batch_size, replace=True, p=probs)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        self._last_indices = indices
        self._last_weights = weights.astype(np.float32)
        return self._get_samples(indices, env=env)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = np.abs(td_errors) + 1e-6
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max())


class DuelingDoubleDQNPER(DQN):
    def __init__(self, *args, per_beta_start: float = 0.4, **kwargs):
        kwargs["policy"] = DuelingCnnPolicy
        kwargs["replay_buffer_class"] = PrioritizedReplayBuffer
        kwargs["replay_buffer_kwargs"] = {"alpha": 0.6}
        super().__init__(*args, **kwargs)
        self.per_beta_start = per_beta_start


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agent_filepath", required=True,
                    help="path to model zip (omit the .zip extension)")
parser.add_argument("-e", "--num_episodes", type=int, default=3,
                    help="number of episodes to record (default 3)")
parser.add_argument("-o", "--output_dir", default="videos",
                    help="directory to save videos (default: videos/)")
parser.add_argument("-n", "--name", default="dueling_per",
                    help="filename prefix for saved videos (default: dueling_per)")
parser.add_argument("-f", "--frameskip", type=int, default=4)
parser.add_argument("-r", "--repeat_action_probability", type=float, default=0.25)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

model = DuelingDoubleDQNPER.load(
    args.agent_filepath,
    custom_objects={
        "replay_buffer_class": PrioritizedReplayBuffer,
        "policy_class": DuelingCnnPolicy,
    },
)

for ep in range(args.num_episodes):
    env = Monitor(gym.make(
        "ALE/Pacman-v5",
        render_mode="rgb_array",
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
    ))

    frames = []

    def capture(locals_, globals_):
        frames.append(env.render())

    mean_rwd, _ = evaluate_policy(
        model.policy, env,
        n_eval_episodes=1,
        deterministic=True,
        callback=capture,
    )

    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=args.fps)
        out_path = os.path.join(args.output_dir, f"{args.name}_ep{ep+1:02d}_rwd{int(mean_rwd)}.mp4")
        clip.write_videofile(out_path, logger=None)
        print(f"[ep {ep+1}/{args.num_episodes}] reward={mean_rwd:.0f}  saved → {out_path}")
    except ImportError:
        print("moviepy not found. Install with: pip install moviepy")
        break

    env.close()
