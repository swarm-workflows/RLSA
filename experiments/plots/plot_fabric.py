"""FABRIC real-WAN figures: ``fig_fabric_robustness`` and ``fig_fabric_scaling``.

Each JSON cell is one (mode, algo, N, loss, seed) run written by
``experiments.fabric.run_fabric``. Lines are means over seeds, error bars +-1 std.
Also prints the scaling table (Table III).

    python -m experiments.plots.plot_fabric --robustness-dir results/fabric/robustness \
        --scaling-dir results/fabric/scaling --fig-dir figures
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.plots.style import apply as apply_style
from experiments.plots.style import label as label_for
from experiments.plots.style import method_color, pct, save


def load(dir_: str) -> list[dict]:
    """Glob ``*.json`` under ``dir_`` and return parsed dicts.

    Files that fail to parse are skipped with a warning rather than crashing.
    """
    rows = []
    for f in sorted(glob.glob(str(Path(dir_) / "*.json"))):
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception as e:  # malformed / unreadable JSON
            print(f"  [warn] skipping {f}: {e}")
    return rows


def algos_in(rows: list[dict]) -> list[str]:
    """Distinct algo names present in ``rows``, in first-seen order."""
    seen: list[str] = []
    for r in rows:
        a = r.get("algo")
        if a is not None and a not in seen:
            seen.append(a)
    return seen


def agg(rows: list[dict], xkey: str, algo: str, metric: str):
    """Aggregate one algo's rows by ``xkey``: mean/std of ``metric`` per x.

    Returns (xs, means, stds) sorted by x. Rows missing either ``xkey`` or
    ``metric`` are skipped, so a metric absent everywhere yields empty arrays.
    """
    buckets: dict[float, list[float]] = {}
    for r in rows:
        if r.get("algo") != algo:
            continue
        if xkey not in r or metric not in r:
            continue
        x = r[xkey]
        v = r[metric]
        if x is None or v is None:
            continue
        buckets.setdefault(x, []).append(float(v))
    xs = sorted(buckets)
    means = [float(np.mean(buckets[x])) for x in xs]
    stds = [float(np.std(buckets[x])) for x in xs]
    return xs, means, stds


def _color(algo: str) -> str:
    """Paper-wide color for an algo, gracefully defaulting if unknown."""
    return method_color(algo, default=None)


def _panel(ax, rows, xkey, metric, xlabel, ylabel, title):
    """Draw one metric-vs-xkey panel, one line per algo present."""
    drawn = False
    for algo in algos_in(rows):
        xs, means, stds = agg(rows, xkey, algo, metric)
        if not xs:  # this algo has no data for this metric
            continue
        ax.errorbar(xs, means, yerr=stds, marker="o",
                    label=label_for(algo), color=_color(algo), capsize=3)
        drawn = True
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if drawn:
        ax.legend(fontsize=8)
    return drawn


def plot_robustness(rows: list[dict], out_pdf: Path) -> None:
    """3 panels (makespan, Jain, LLM/job) vs netem_loss_pct."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    panels = [
        ("makespan", "Makespan (s)", "Makespan"),
        ("jain_fairness", "Jain fairness", "Fairness"),
        ("llm_calls_per_job", "LLM/job", "LLM calls/job"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        _panel(ax, rows, "netem_loss_pct", metric,
               f"Network loss ({pct()})", ylabel, title)
    fig.suptitle("FABRIC real-WAN robustness to packet loss", fontsize=11)
    fig.tight_layout()
    save(fig, out_pdf)
    plt.close(fig)


def plot_scaling(rows: list[dict], out_pdf: Path) -> None:
    """3 panels (makespan, round walltime, WAN RTT p95) vs n_agents."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    panels = [
        ("makespan", "Makespan (s)", "Makespan"),
        ("round_walltime_ms_mean", "Round walltime (ms)", "Round walltime (mean)"),
        ("wan_rtt_ms_p95", "WAN RTT p95 (ms)", "WAN RTT (p95)"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        _panel(ax, rows, "n_agents", metric,
               "Number of agents N", ylabel, title)
    fig.suptitle("FABRIC real-WAN scaling vs N", fontsize=11)
    fig.tight_layout()
    save(fig, out_pdf)
    plt.close(fig)


def print_scaling_table(rows: list[dict]) -> None:
    """Table III: mean over seeds per (algo, N)."""
    print(f"\n{'N':>3} {'algo':<12} {'Jain':>6} {'LLM/j':>6} {'Round (ms)':>11}")
    for algo in algos_in(rows):
        for n in sorted({r["n_agents"] for r in rows if r.get("algo") == algo}):
            sub = [r for r in rows if r.get("algo") == algo and r["n_agents"] == n]
            mean = lambda k: float(np.mean([r[k] for r in sub]))
            print(f"{n:>3} {label_for(algo):<12} {mean('jain_fairness'):6.3f} "
                  f"{mean('llm_calls_per_job'):6.2f} {mean('round_walltime_ms_mean'):11.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness-dir", default="results/fabric/robustness")
    ap.add_argument("--scaling-dir", default="results/fabric/scaling")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--table-only", action="store_true", help="only print Table III")
    args = ap.parse_args()

    scaling = load(args.scaling_dir)
    if args.table_only:
        print_scaling_table(scaling)
        return

    apply_style()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    robustness = load(args.robustness_dir)
    if robustness:
        plot_robustness(robustness, fig_dir / "fig_fabric_robustness.pdf")
    else:
        print(f"[skip] no robustness cells in {args.robustness_dir}")
    if scaling:
        plot_scaling(scaling, fig_dir / "fig_fabric_scaling.pdf")
        print_scaling_table(scaling)
    else:
        print(f"[skip] no scaling cells in {args.scaling_dir}")


if __name__ == "__main__":
    main()
