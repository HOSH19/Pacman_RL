import os
import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="tab10")
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444466",
    "grid.color":        "#2a2a4a",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#444466",
})

# ---------------------------------------------------------------------------
# CONFIGURE THIS
# ---------------------------------------------------------------------------

ENTITY  = "hoshuhan-university-college-london-ucl-"
PROJECT = "ale-pacman-v5"

RUNS = {
    # "Dueling DDQN":     "ykmc0k4e",
    "Dueling DDQN":     "za54647c",
    "Dueling DDQN+PER": "z45h4kd8",
    "DDQN (baseline)": "gm9420ue",
}

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", default="charts/wandb_analysis",
                    help="directory to save all outputs (default: charts/wandb_analysis)")
parser.add_argument("--samples", type=int, default=5000,
                    help="history rows to fetch per run (default: 5000)")
args = parser.parse_args()

OUT = args.output_dir
os.makedirs(OUT, exist_ok=True)
print(f"Output → {OUT}/\n")


# ---------------------------------------------------------------------------
# Pull history
# ---------------------------------------------------------------------------

api = wandb.Api()
histories = {}

for label, run_id in RUNS.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df  = run.history(samples=args.samples)
    df["run"] = label
    histories[label] = df
    print(f"{label}: {len(df)} rows")

all_runs = pd.concat(histories.values(), ignore_index=True)

metric_cols = sorted(c for c in all_runs.columns if c not in ("_step", "_runtime", "_timestamp", "run"))
print("\nAvailable metrics:")
print("\n".join(f"  {c}" for c in metric_cols))
print()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def plot_metric(df, metric, title, ylabel, filename, smooth=0.0, ymin=None, ymax=None):
    x_col = "global_step" if "global_step" in df.columns else "_step"
    sub = df[[x_col, "run", metric]].dropna()
    if sub.empty:
        print(f"  [skip] {metric} — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, grp in sub.groupby("run"):
        x = grp[x_col]
        y = grp[metric]
        if smooth > 0:
            y = y.ewm(alpha=1 - smooth).mean()
        ax.plot(x, y, label=label, linewidth=1.6)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

print("Generating plots...")

plot_metric(all_runs, "rollout/ep_rew_mean",
            "Training Reward vs Steps", "Mean Episode Reward",
            "01_train_reward.png", smooth=0.9)

plot_metric(all_runs, "eval/mean_reward",
            "Evaluation Reward vs Steps", "Mean Eval Reward (deterministic)",
            "02_eval_reward.png")

plot_metric(all_runs, "rollout/ep_len_mean",
            "Episode Length vs Steps", "Mean Episode Length (frames)",
            "03_ep_length.png", smooth=0.9)

plot_metric(all_runs, "train/loss",
            "Training Loss vs Steps", "Smooth L1 Loss",
            "04_loss.png", smooth=0.9)

plot_metric(all_runs, "rollout/exploration_rate",
            "Epsilon (Exploration Rate) vs Steps", "Epsilon",
            "05_epsilon.png", ymin=0, ymax=1)

plot_metric(all_runs, "per/beta",
            "PER Beta Annealing vs Steps", "Beta",
            "06_per_beta.png", ymin=0, ymax=1)

plot_metric(all_runs, "per/max_priority",
            "PER Max Priority vs Steps", "Max Priority",
            "07_per_max_priority.png", smooth=0.9)

# Eval reward box plot
eval_data = all_runs[["run", "eval/mean_reward"]].dropna()
if not eval_data.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=eval_data, x="run", y="eval/mean_reward", ax=ax)
    ax.set_title("Eval Reward Distribution Across Checkpoints")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Eval Reward")
    fig.tight_layout()
    path = os.path.join(OUT, "08_eval_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\nBuilding summary table...")

rows = []
for label, df in histories.items():
    row = {"Run": label}
    for col, alias in [
        ("rollout/ep_rew_mean", "Train Reward (final 10%)"),
        ("eval/mean_reward",    "Eval Reward (best)"),
        ("eval/mean_reward",    "Eval Reward (final 10%)"),
        ("train/loss",         "Loss (final 10%)"),
    ]:
        if col not in df.columns:
            row[alias] = float("nan")
            continue
        vals = df[col].dropna()
        if "best" in alias:
            row[alias] = vals.max()
        else:
            tail = vals.iloc[int(len(vals) * 0.9):]
            row[alias] = tail.mean()
    rows.append(row)

summary = pd.DataFrame(rows).set_index("Run").round(1)
print(summary.to_string())
csv_path = os.path.join(OUT, "summary_table.csv")
summary.to_csv(csv_path)
print(f"\n  saved → {csv_path}")


# ---------------------------------------------------------------------------
# Raw history CSVs
# ---------------------------------------------------------------------------

print("\nExporting raw histories...")
for label, df in histories.items():
    slug = label.lower().replace(" ", "_").replace("+", "_plus_")
    path = os.path.join(OUT, f"history_{slug}.csv")
    df.to_csv(path, index=False)
    print(f"  saved → {path}")

print("\nDone.")
