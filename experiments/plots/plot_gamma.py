"""Discount-factor ablation in the coupled auction (paper figure ``fig_gamma_ablation``).

    python -m experiments.plots.plot_gamma --results results/auction/gamma --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import load_json_glob, mean_std
from experiments.plots.style import COLORS, CYCLE, save
from experiments.plots.style import apply as apply_style

GAMMAS = ["0.0", "0.95", "0.99"]
METRICS = [
    ("makespan", "Makespan (s)", "(a) Makespan"),
    ("jain_fairness", "Jain fairness", "(b) Fairness"),
    ("llm_calls_per_job", "LLM calls / job", "(c) LLM cost"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/auction/gamma")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    data = {}
    for g in GAMMAS:
        ds = load_json_glob(Path(args.results) / f"g{g}_rlsa_N4_seed*.json")
        if ds:
            data[g] = {m: mean_std(ds, m) for m, _, _ in METRICS}
            print(f"gamma={g}: n={len(ds)} " + " ".join(
                f"{m}={data[g][m][0]:.3f}" for m, _, _ in METRICS))
    if not data:
        raise SystemExit(f"no g<gamma>_rlsa_N4_seed*.json cells in {args.results}")

    present = [g for g in GAMMAS if g in data]
    x = np.arange(len(present))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    for ax, (m, ylabel, title) in zip(axes, METRICS):
        ax.bar(x, [data[g][m][0] for g in present], yerr=[data[g][m][1] for g in present],
               color=CYCLE[: len(present)], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\gamma$={g}" for g in present], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if m == "jain_fairness":
            ax.set_ylim(0, 1.05)
            ax.axhline(0.25, color=COLORS["ref"], lw=0.8, ls="--", label="1/N")
            ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out / "fig_gamma_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
