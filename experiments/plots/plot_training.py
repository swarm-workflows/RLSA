"""Training curves in the gating environment (paper figures ``fig_training_curves``
and ``fig_hf_training``).

Four panels, mean +- std over the seed CSVs written by
``experiments.single_agent.train``:

  (a) episode return vs. fixed reference policies
  (b) LLM invocation rate on easy vs. hard jobs
  (c) bid error per difficulty bucket
  (d) reward components R_make / R_util (left axis) and R_cost (right axis)

    python -m experiments.plots.plot_training results/single_agent/training/seed*.csv \
        -o figures/fig_training_curves.pdf
    python -m experiments.plots.plot_training results/single_agent/qwen_model_sweep/qwen2.5-3b/seed*.csv \
        -o figures/fig_hf_training.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.plots.style import COLORS, pct, save
from experiments.plots.style import apply as apply_style

# Per-step reward of fixed reference policies in the mock gating env
# (deterministic evaluation on a fresh seed), scaled to a 100-step episode.
REF_SELECTIVE_ORACLE = -1.135
REF_ALWAYS_SURROGATE = -1.340
REF_ALWAYS_LLM = -1.771


def _load_runs(csv_paths: List[str]) -> pd.DataFrame:
    """Stack per-seed CSVs into a long-form dataframe with a `seed` column."""
    frames = []
    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        df["seed"] = i
        df["source"] = Path(p).name
        frames.append(df)
    if not frames:
        raise SystemExit("No CSVs found.")
    return pd.concat(frames, ignore_index=True)


def _band(ax, df: pd.DataFrame, x: str, y: str, label: str, color: str,
          smooth: int = 1) -> None:
    """Plot mean ± std band of column `y` over runs, indexed by `x`."""
    pivot = df.pivot(index=x, columns="seed", values=y)
    if smooth > 1:
        pivot = pivot.rolling(window=smooth, min_periods=1).mean()
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)
    ax.plot(pivot.index, mean.values, color=color, label=label, linewidth=2)
    if pivot.shape[1] > 1:
        ax.fill_between(pivot.index, (mean - std).values, (mean + std).values,
                        color=color, alpha=0.18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", help="One or more per-seed CSVs.")
    p.add_argument("-o", "--out", default="figures/fig_training_curves.pdf",
                   help="output figure (.pdf or .png)")
    p.add_argument("--x", choices=["update", "global_step"], default="global_step")
    p.add_argument("--smooth", type=int, default=3,
                   help="Rolling-mean window (in PPO updates) for smoothing.")
    p.add_argument("--no-refs", action="store_true",
                   help="omit the mock-env reference lines in panel (a)")
    args = p.parse_args()

    df = _load_runs(args.csv)
    n_seeds = df["seed"].nunique()
    # Truncate to the shortest run so the pivot doesn't introduce NaNs for
    # runs that are still in progress.
    min_updates = int(df.groupby("seed")["update"].max().min())
    df = df[df["update"] <= min_updates]
    print(f"Loaded {len(args.csv)} CSV(s), {n_seeds} seed(s), "
          f"{min_updates} updates each (truncated to common length).")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)

    # -------- (a) Episode return --------
    # Episodes are 100 steps long, so the rolling "ep_return_10" is about
    # 100 x the per-step reward.
    ax = axes[0, 0]
    _band(ax, df, args.x, "ep_return_10",
          label="RLSA (rolling mean of 10 episodes)",
          color=COLORS["primary"], smooth=args.smooth)
    if not args.no_refs:
        ax.axhline(REF_SELECTIVE_ORACLE * 100, color=COLORS["green"], linestyle="--", linewidth=1,
                   label="Selective oracle (privileged)")
        ax.axhline(REF_ALWAYS_SURROGATE * 100, color=COLORS["purple"], linestyle=":", linewidth=1,
                   label="Always-surrogate (oracle bid)")
        ax.axhline(REF_ALWAYS_LLM * 100, color=COLORS["accent"], linestyle=":", linewidth=1,
                   label="Always-LLM")
    ax.set_xlabel("PPO update" if args.x == "update" else "Environment steps")
    ax.set_ylabel("Episode return (100-step rolling mean)")
    ax.set_title("(a) Episode return vs baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # -------- (b) Gate% per bucket --------
    ax = axes[0, 1]
    df_pct = df.copy()
    df_pct["gate_easy_pct"] = 100.0 * df_pct["gate_easy"]
    df_pct["gate_hard_pct"] = 100.0 * df_pct["gate_hard"]
    _band(ax, df_pct, args.x, "gate_hard_pct",
          label="Hard jobs (gate=1 desired)", color=COLORS["accent"], smooth=args.smooth)
    _band(ax, df_pct, args.x, "gate_easy_pct",
          label="Easy jobs (gate=0 desired)", color=COLORS["green"], smooth=args.smooth)
    ax.axhline(100, color=COLORS["muted"], linestyle=":", linewidth=0.8)
    ax.axhline(0, color=COLORS["muted"], linestyle=":", linewidth=0.8)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("PPO update" if args.x == "update" else "Environment steps")
    ax.set_ylabel(f"Pr(invoke LLM)  [{pct()}]")
    ax.set_title("(b) Selective gating emerges without supervision")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    # -------- (c) Bid score error per bucket --------
    ax = axes[1, 0]
    _band(ax, df, args.x, "score_err_hard",
          label="Hard jobs", color=COLORS["accent"], smooth=args.smooth)
    _band(ax, df, args.x, "score_err_easy",
          label="Easy jobs", color=COLORS["green"], smooth=args.smooth)
    ax.set_xlabel("PPO update" if args.x == "update" else "Environment steps")
    ax.set_ylabel("Mean |bid - true bid| / 100")
    ax.set_title("(c) Bid score error")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # -------- (d) Reward components --------
    # R_cost can be ~100x smaller than R_make (real-LLM runs), so it gets a
    # twin right axis.
    ax = axes[1, 1]
    if "r_make_mean" in df.columns:
        _band(ax, df, args.x, "r_make_mean",
              label=r"$R_{\mathrm{make}}$ (makespan)", color=COLORS["primary"], smooth=args.smooth)
        _band(ax, df, args.x, "r_util_mean",
              label=r"$R_{\mathrm{util}}$ (utilization)", color=COLORS["green"], smooth=args.smooth)
        ax.axhline(0, color=COLORS["muted"], linestyle=":", linewidth=0.5)
        ax.set_xlabel("PPO update" if args.x == "update" else "Environment steps")
        ax.set_ylabel("Per-step reward contribution")
        ax.set_title("(d) Reward components (pre-$\\lambda$ weighting)")
        ax.grid(alpha=0.3)

        ax_cost = ax.twinx()
        _band(ax_cost, df, args.x, "r_cost_mean",
              label=r"$R_{\mathrm{cost}}$ (LLM penalty, right axis)",
              color=COLORS["accent"], smooth=args.smooth)
        ax_cost.set_ylabel(r"$R_{\mathrm{cost}}$ (right axis)",
                           color=COLORS["accent"])
        ax_cost.tick_params(axis="y", labelcolor=COLORS["accent"])

        # Merge legends from both axes into one box.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_cost.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    else:
        # CSVs without reward components: show losses instead.
        df_scaled = df.copy()
        df_scaled["loss_imitation_10x"] = 10.0 * df_scaled["loss_imitation"]
        _band(ax, df, args.x, "loss_policy",
              label="Policy loss", color=COLORS["primary"], smooth=args.smooth)
        _band(ax, df_scaled, args.x, "loss_imitation_10x",
              label="Imitation loss (x10)", color=COLORS["purple"], smooth=args.smooth)
        ax.axhline(0, color=COLORS["muted"], linestyle=":", linewidth=0.5)
        ax.set_xlabel("PPO update" if args.x == "update" else "Environment steps")
        ax.set_ylabel("Loss")
        ax.set_title("(d) Loss components")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out_path)


if __name__ == "__main__":
    main()
