"""Auction figures: ``fig_distributed_main``, ``fig_scale_n``, ``fig_robustness_drop``.

    python -m experiments.plots.plot_auction --results results/auction --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import load_json_glob, mean_std
from experiments.plots.style import COLORS, label, method_color, save
from experiments.plots.style import apply as apply_style

ALGOS_MOCK = ["random", "capacity", "always_surrogate", "always_llm", "oracle", "rlsa"]
ALGOS_HF = ["always_llm", "oracle", "rlsa"]


def _gather(run_dir: Path, algos, n=4):
    rows = []
    for a in algos:
        ds = load_json_glob(run_dir / f"{a}_N{n}_seed*.json")
        if not ds:
            continue
        row = {"algo": a}
        for key, name in (("makespan", "mksp"), ("jain_fairness", "jain"), ("llm_calls_per_job", "llm")):
            row[name], row[name + "_std"] = mean_std(ds, key)
        rows.append(row)
    return rows


def fig_distributed_main(results: Path, out: Path) -> None:
    """Headline N=4 auction: mock (left column) and real Qwen-7B (middle column)."""
    mock = _gather(results / "headline_mock", ALGOS_MOCK)
    hf = _gather(results / "headline_qwen7b", ALGOS_HF)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.0))

    def bars(ax, rows, key, title, ylabel):
        x = np.arange(len(rows))
        ax.bar(x, [r[key] for r in rows], yerr=[r[key + "_std"] for r in rows],
               color=[method_color(r["algo"]) for r in rows], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([label(r["algo"]) for r in rows], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    for col, (rows, title) in enumerate([(mock, "Mock LLM (N=4, 400 jobs)"),
                                         (hf, "Real Qwen-7B (N=4, 200 jobs)")]):
        bars(axes[0, col], rows, "mksp", f"Makespan - {title}", "Makespan (s)")
        ax = axes[1, col]
        bars(ax, rows, "jain", f"Jain - {title}", "Jain fairness")
        ax.axhline(0.25, color=COLORS["ref"], lw=0.8, ls="--", label="1/N=0.25")
        ax.set_ylim(0, 1.05)
        if col == 0:
            ax.legend(fontsize=8)
    bars(axes[0, 2], mock, "llm", "LLM calls/job - Mock", "LLM/job")
    bars(axes[1, 2], hf, "llm", "LLM calls/job - Real Qwen-7B", "LLM/job")
    fig.tight_layout()
    save(fig, out / "fig_distributed_main.pdf")
    plt.close(fig)


def fig_scale_n(results: Path, out: Path) -> None:
    """Makespan and Jain fairness vs. the number of bidders."""
    ns = [1, 2, 4, 8, 16]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for algo in ("rlsa", "always_llm", "always_surrogate"):
        xs, m, ms, j, js = [], [], [], [], []
        for n in ns:
            ds = load_json_glob(results / "scale_n" / f"scale_{algo}_N{n}_seed*.json")
            if not ds:
                continue
            mm, mstd = mean_std(ds, "makespan")
            jm, jstd = mean_std(ds, "jain_fairness")
            xs.append(n); m.append(mm); ms.append(mstd); j.append(jm); js.append(jstd)
        if xs:
            kw = dict(marker="o", label=label(algo), color=method_color(algo), capsize=3)
            ax1.errorbar(xs, m, yerr=ms, **kw)
            ax2.errorbar(xs, j, yerr=js, **kw)
    ax2.plot(ns, [1.0 / n for n in ns], color=COLORS["muted"], lw=0.8, ls=":",
             label="1/N (det. tiebreak)")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels(ns)
        ax.set_xlabel("Number of agents N")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Throughput vs N")
    ax1.legend(fontsize=8)
    ax2.set_yscale("log")
    ax2.set_yticks([1, 0.5, 0.25, 0.125, 0.0625])
    ax2.set_yticklabels(["1", "1/2", "1/4", "1/8", "1/16"])
    ax2.set_ylabel("Jain fairness")
    ax2.set_title("(b) Fairness vs N")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, out / "fig_scale_n.pdf")
    plt.close(fig)


def fig_robustness_drop(results: Path, out: Path) -> None:
    """Message-drop sweep, RLSA retrained at each drop level."""
    tags = ["0.0", "0.05", "0.10", "0.20"]  # filename tags as written by the sweep
    drops = [float(t) for t in tags]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    metrics = [("makespan", "Makespan", "Makespan (s)"),
               ("jain_fairness", "Jain fairness", "Jain"),
               ("llm_calls_per_job", "LLM calls/job", "LLM/job")]
    for algo in ("rlsa", "always_llm"):
        data = {m: ([], [], []) for m, _, _ in metrics}
        for tag, dr in zip(tags, drops):
            ds = load_json_glob(results / "drop_retrain" / f"drop{tag}_{algo}_N4_seed*.json")
            if not ds:
                continue
            for m, _, _ in metrics:
                mu, sd = mean_std(ds, m)
                data[m][0].append(dr); data[m][1].append(mu); data[m][2].append(sd)
        for ax, (m, _, _) in zip(axes, metrics):
            if data[m][0]:
                ax.errorbar(*data[m][:2], yerr=data[m][2], marker="o", label=label(algo),
                            color=method_color(algo), capsize=3)
    for ax, (_, title, ylabel) in zip(axes, metrics):
        ax.set_xlabel("Net drop probability")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "fig_robustness_drop.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/auction")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    fig_distributed_main(results, out)
    fig_scale_n(results, out)
    fig_robustness_drop(results, out)


if __name__ == "__main__":
    main()
