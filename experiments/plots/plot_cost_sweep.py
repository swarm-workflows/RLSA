"""Cost-weight sweep in the gating environment (paper figure ``fig_cost_sweep``).

(a) quality reward (R_make + R_util) vs. LLM call rate as beta varies, with the
    fixed reference policies; (b) easy/hard gate rate vs. beta.

    python -m experiments.plots.plot_cost_sweep results/single_agent/cost_sweep/sweep.csv \
        -o figures/fig_cost_sweep.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.plots.style import COLORS, pct, save
from experiments.plots.style import apply as apply_style

# Fixed reference policies in the mock gating env: LLM call rate and per-step
# quality reward (composite reward minus the cost term at beta=1), from a
# deterministic evaluation on a fresh seed.
REF_POINTS = {
    "Selective oracle":  {"gate": 0.312, "R": -0.823, "marker": "*", "color": COLORS["amber"]},
    "Always-surrogate":  {"gate": 0.000, "R": -1.340, "marker": "s", "color": COLORS["green"]},
    "Always-LLM":        {"gate": 1.000, "R": -0.778, "marker": "X", "color": COLORS["accent"]},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="results/single_agent/cost_sweep/sweep.csv")
    p.add_argument("-o", "--out", default="figures/fig_cost_sweep.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['cost_beta'].nunique()} beta values.")

    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    y_col = "quality_R" if "quality_R" in df.columns else "overall_R"
    print(f"  using y axis = {y_col}")
    agg = df.groupby("cost_beta").agg(
        R_mean=(y_col, "mean"),
        R_std=(y_col, "std"),
        gate_mean=("gate_overall", "mean"),
        gate_std=("gate_overall", "std"),
        easy_mean=("gate_easy", "mean"),
        easy_std=("gate_easy", "std"),
        hard_mean=("gate_hard", "mean"),
        hard_std=("gate_hard", "std"),
    ).reset_index()

    # ---------------- (a) Pareto curve ----------------
    ax = axes[0]
    xs = 100 * agg["gate_mean"].values
    ys = agg["R_mean"].values
    betas = agg["cost_beta"].values
    ax.plot(xs, ys, "-", color=COLORS["primary"], linewidth=1.5, zorder=2,
            label=r"RLSA (sweep over $\lambda_{cost}\cdot\beta$)")
    ax.errorbar(xs, ys, xerr=100 * agg["gate_std"].values,
                yerr=agg["R_std"].values, fmt="none",
                ecolor=COLORS["muted"], alpha=0.6, capsize=3, zorder=1)
    sc = ax.scatter(xs, ys, c=betas, cmap="viridis", s=70, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"cost weight $\beta$")
    for name, ref in REF_POINTS.items():
        ax.scatter(
            100 * ref["gate"], ref["R"],
            marker=ref["marker"], color=ref["color"], s=130, zorder=5,
            label=name,
        )
    ax.set_xlabel(f"LLM call rate [{pct()}]")
    ax.set_ylabel(r"Quality reward $R_{\mathrm{make}} + R_{\mathrm{util}}$ (excl. cost)")
    ax.set_title("(a) Pareto frontier: quality vs LLM usage")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-5, 105)

    # ---------------- (b) Gate% per bucket vs beta ----------------
    ax = axes[1]
    ax.errorbar(
        agg["cost_beta"], 100 * agg["hard_mean"],
        yerr=100 * agg["hard_std"], fmt="o-",
        color=COLORS["accent"], linewidth=2, markersize=6, capsize=3,
        label="Hard jobs",
    )
    ax.errorbar(
        agg["cost_beta"], 100 * agg["easy_mean"],
        yerr=100 * agg["easy_std"], fmt="o-",
        color=COLORS["green"], linewidth=2, markersize=6, capsize=3,
        label="Easy jobs",
    )
    ax.set_xscale("symlog", linthresh=0.25)
    ax.axhline(100, color=COLORS["muted"], linestyle=":", linewidth=0.6)
    ax.axhline(0, color=COLORS["muted"], linestyle=":", linewidth=0.6)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"LLM cost weight  $\lambda_{cost}\cdot\beta$")
    ax.set_ylabel(f"Pr(invoke LLM)  [{pct()}]")
    ax.set_title("(b) Per-bucket gating vs cost weight")
    ax.legend(loc="center right", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out_path)


if __name__ == "__main__":
    main()
