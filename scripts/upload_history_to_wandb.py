"""
Upload the full DQN v2 training history to Weights & Biases.

Each dqn_v2-x agent directory has an evaluations.npz file with:
  - timesteps: (N,) array of env steps at each eval checkpoint
  - results:   (N, 10) array of episode rewards (10 eval episodes per checkpoint)

The runs were trained sequentially, with each run continuing from where
the last left off, so the timesteps are already on a global scale and
can be stitched directly without an offset.

Usage:
    python upload_history_to_wandb.py
    python upload_history_to_wandb.py --project my-project --run-name dqn-v2-history
"""

import argparse
import os
import numpy as np
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="ale-pacman-v5", help="W&B project name")
parser.add_argument("--run-name", default="dqn-v2-full-history", help="W&B run name")
parser.add_argument("--agents-dir", default="agents", help="Path to agents directory")
args = parser.parse_args()

# Collect all dqn_v2 evaluation files in order
agent_dirs = sorted(
    d for d in os.listdir(args.agents_dir)
    if d.startswith("dqn_v2")
)

all_timesteps = []
all_means = []
all_stds = []
all_mins = []
all_maxs = []
run_boundaries = []  # (timestep, run_name) for vertical markers

for agent_dir in agent_dirs:
    path = os.path.join(args.agents_dir, agent_dir, "evaluations.npz")
    if not os.path.exists(path):
        print(f"  Skipping {agent_dir} — no evaluations.npz")
        continue

    data = np.load(path)
    timesteps = data["timesteps"]       # (N,)
    results = data["results"]           # (N, 10)

    means = results.mean(axis=1)
    stds = results.std(axis=1)
    mins = results.min(axis=1)
    maxs = results.max(axis=1)

    all_timesteps.extend(timesteps.tolist())
    all_means.extend(means.tolist())
    all_stds.extend(stds.tolist())
    all_mins.extend(mins.tolist())
    all_maxs.extend(maxs.tolist())
    run_boundaries.append((int(timesteps[0]), agent_dir))

    print(f"  {agent_dir}: {len(timesteps)} checkpoints, "
          f"steps {timesteps[0]:,}–{timesteps[-1]:,}, "
          f"mean score {means.mean():.1f} ± {stds.mean():.1f}")

print(f"\nTotal checkpoints: {len(all_timesteps)}")
print(f"Timestep range: {all_timesteps[0]:,} – {all_timesteps[-1]:,}")

run = wandb.init(
    project=args.project,
    name=args.run_name,
    config={
        "algorithm": "DQN",
        "environment": "ALE/Pacman-v5",
        "total_timesteps": all_timesteps[-1],
        "num_runs": len(agent_dirs),
        "eval_episodes_per_checkpoint": 10,
    },
)

for i, timestep in enumerate(all_timesteps):
    log = {
        "timestep": timestep,
        "eval/mean_reward": all_means[i],
        "eval/std_reward": all_stds[i],
        "eval/min_reward": all_mins[i],
        "eval/max_reward": all_maxs[i],
        "eval/mean_plus_std": all_means[i] + all_stds[i],
        "eval/mean_minus_std": all_means[i] - all_stds[i],
    }
    # Tag which training run version this checkpoint belongs to
    for j, (boundary_ts, run_name) in enumerate(run_boundaries):
        next_ts = run_boundaries[j + 1][0] if j + 1 < len(run_boundaries) else float("inf")
        if boundary_ts <= timestep < next_ts:
            log["run_version"] = run_name
            break

    wandb.log(log, step=timestep)

wandb.finish()
print(f"\nDone. View at: {run.url}")
