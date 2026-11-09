"""Heterogeneous LLM bidders (paper figure ``fig_hetero_llm``).

Agents 0-3 run Qwen-2.5 3B / 7B / 14B and the mock oracle. Plots per-agent wins
and per-agent deadline misses for Always-LLM vs. RLSA, averaged over repeats.

    python -m experiments.plots.plot_hetero --results results/auction/hetero_llm --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import load_json_glob
from experiments.plots.style import COLORS, save
from experiments.plots.style import apply as apply_style

AGENT_LABELS = ["3B", "7B", "14B", "mock"]


def per_agent(results: Path, algo: str, key: str):
    rows = load_json_glob(results / f"trueHetero_seed*_{algo}_N4_seed*.json")
    if not rows:
        raise SystemExit(f"no trueHetero_*_{algo}_N4 cells in {results}")
    arr = np.array([[r[key].get(str(i), 0) for i in range(4)] for r in rows])
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/auction/hetero_llm")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    x, w = np.arange(4), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    wins_ll, wins_ll_sd = per_agent(results, "always_llm", "wins_per_agent")
    wins_rl, wins_rl_sd = per_agent(results, "rlsa", "wins_per_agent")
    ax1.bar(x - w / 2, wins_ll, w, yerr=wins_ll_sd, label="Always-LLM", color=COLORS["accent"], capsize=3)
    ax1.bar(x + w / 2, wins_rl, w, yerr=wins_rl_sd, label="RLSA", color=COLORS["primary"], capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(AGENT_LABELS)
    ax1.set_xlabel("Per-agent bidder")
    ax1.set_ylabel("Mean wins (of 100 jobs)")
    ax1.set_title("(a) Workload distribution across heterogeneous bidders", fontsize=9)
    ax1.legend(fontsize=9)

    miss_ll, _ = per_agent(results, "always_llm", "deadline_misses_per_agent")
    miss_rl, _ = per_agent(results, "rlsa", "deadline_misses_per_agent")
    ax2.bar(x - w / 2, miss_ll, w, label="Always-LLM", color=COLORS["accent"])
    ax2.bar(x + w / 2, miss_rl, w, label="RLSA", color=COLORS["primary"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(AGENT_LABELS)
    ax2.set_xlabel("Per-agent bidder")
    ax2.set_ylabel("Deadline misses (of 100 rounds)")
    ax2.set_title("(b) Heterogeneous latency vs deadline", fontsize=9)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    save(fig, out / "fig_hetero_llm.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
