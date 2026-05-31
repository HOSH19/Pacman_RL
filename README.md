# ALE/Pacman-v5 — Dueling DDQN + Prioritized Experience Replay

Building upon [ledmands/ALE-Pacman-v5](https://huggingface.co/ledmands/ALE-Pacman-v5), this project extends the original DQN implementation with three progressive improvements — Double DQN, Dueling architecture, and Prioritized Experience Replay — trained using [Stable Baselines3](https://stable-baselines3.readthedocs.io/) on `ALE/Pacman-v5`.

---

## Demo

https://github.com/user-attachments/assets/34c8a46f-d757-4d69-86de-2050097e27ba

*Dueling DDQN+PER best model checkpoint — eval reward: 436. DDQN and Dueling DDQN demos coming once baseline runs complete.*

---

## Results

| Algorithm | Best Eval Reward | Final Eval Reward (avg, last 10%) | Final Loss |
|---|---|---|---|
| DDQN (baseline) | 134.3 | 113.3 | 0.2 |
| Dueling DDQN | 252.4 | 196.3 | 0.3 |
| **Dueling DDQN + PER** | **415.4** | **370.0** | **0.1** |

---

## Key Findings

### Learning Curves
![Training Reward](charts/wandb_analysis/01_train_reward.png)
![Eval Reward](charts/wandb_analysis/02_eval_reward.png)

Each improvement compounds: DDQN plateaus ~100, Dueling DDQN reaches ~250 before becoming unstable, PER climbs steadily to ~415. PER's final-10% eval mean (370) stays near its peak while Dueling DDQN degrades (196) and DDQN barely improves (113) — PER stabilises late-stage training rather than just accelerating early learning.

### Training Loss
![Loss](charts/wandb_analysis/04_loss.png)

PER converges to 4× lower final loss (0.1 vs 0.3). Prioritised sampling surfaces the most informative transitions, driving more consistent gradient updates.

### Eval Reward Distribution
![Eval Boxplot](charts/wandb_analysis/08_eval_boxplot.png)

Across all evaluation checkpoints, PER has the highest median and tightest spread. DDQN is concentrated at the bottom with little variance. Dueling DDQN's wide distribution confirms the instability seen in the learning curves.

**Full analysis with all 8 charts → [ANALYSIS.md](ANALYSIS.md)**

---

## Algorithms

### 1. Double DQN (DDQN)
Decouples action selection from evaluation to remove Q-value overestimation bias:
- **Online network** selects: `a* = argmax_a Q_online(s', a)`
- **Target network** scores: `target = r + γ · Q_target(s', a*)`

### 2. Dueling DDQN
Splits the Q-network into two streams after the CNN backbone:
- **Value stream** V(s) — how good is this state
- **Advantage stream** A(s, a) — relative value of each action
- Combined: `Q(s, a) = V(s) + A(s, a) − mean_a(A(s, a))`

### 3. Dueling DDQN + Prioritized Experience Replay (PER)
Samples high-TD-error transitions more frequently:
- Priority ∝ `|TD error|^α` (α = 0.6)
- Importance-sampling weights correct for bias, β anneals 0.4 → 1.0 over training

---

## Repository Structure

```
.
├── notebooks/
│   ├── dqn_pacmanv5_double_dqn.ipynb          # DDQN training (Colab)
│   ├── dqn_pacmanv5_dueling_double.ipynb       # Dueling DDQN training (Colab)
│   └── dqn_pacmanv5_dueling_double_per.ipynb   # Dueling DDQN+PER training (Colab)
├── scripts/
│   ├── train_double_dqn.py        # Train Dueling DDQN locally
│   ├── train_per.py               # Train Dueling DDQN+PER locally
│   ├── evaluate_dueling_per.py    # Evaluate a saved model (prints mean ± std)
│   ├── watch_dueling_per.py       # Watch agent play in real-time
│   ├── record_dueling_per.py      # Record evaluation episodes to .mp4
│   └── wandb_analysis.py          # Pull W&B metrics and generate charts
├── charts/wandb_analysis/         # All plots and CSVs
├── videos/                        # Recorded evaluation episodes
├── ANALYSIS.md                    # Full results and conclusions
└── README.md
```

---

## Setup

```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py wandb moviepy
```

## Usage

```bash
# Evaluate
python scripts/evaluate_dueling_per.py -a checkpoints/best_model -e 10

# Watch live
python scripts/watch_dueling_per.py -a checkpoints/best_model

# Record video
python scripts/record_dueling_per.py -a checkpoints/best_model -e 3 -o videos/

# Re-run analysis
python scripts/wandb_analysis.py -o charts/wandb_analysis
```

---

## Training Setup

All runs: Google Colab T4 GPU, 3,000,000 environment steps.

| Hyperparameter | Value |
|---|---|
| Learning rate | 5e-5 |
| Buffer size | 70,000 |
| Batch size | 64 |
| Gamma | 0.999 |
| Exploration fraction | 0.30 |
| Final epsilon | 0.005 |
| Target update interval | 1,000 steps |
| Learning starts | 100,000 steps |
| PER alpha | 0.6 |
| PER beta start | 0.4 |
| Frameskip | 4 |

---

## Credits

Built upon the original DQN implementation by **Lucas Edmands** — [ledmands/ALE-Pacman-v5](https://huggingface.co/ledmands/ALE-Pacman-v5) on HuggingFace.

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., 2013
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — van Hasselt et al., 2015
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) — Wang et al., 2015
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Schaul et al., 2015
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
