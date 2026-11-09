"""Gating vs. LLM size (paper figure ``fig_model_sweep``).

Reads ``<sweep-dir>/<model-tag>/seed*.csv`` from
``experiments.single_agent.train --bidder hf`` and plots (a) end-of-training
easy/hard gate rates and (b) episode return per model size.

    python -m experiments.plots.plot_model_sweep \
        --sweep-dir results/single_agent/qwen_model_sweep -o figures/fig_model_sweep.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plots.style import COLORS, pct, save
from experiments.plots.style import apply as apply_style


def _param_count_b(tag: str) -> float:
    """Parameter count in billions from a folder tag, e.g. 'qwen2.5-7b' -> 7.0."""
    m = re.search(r"(\d+(?:\.\d+)?)b", tag.lower())
    return float(m.group(1)) if m else float("nan")


def _final_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    return {
        "gate_easy": float(last["gate_easy"]),
        "gate_hard": float(last["gate_hard"]),
        "ep_return_10": float(last["ep_return_10"]),
        "r_make_mean": float(last.get("r_make_mean", float("nan"))),
        "r_util_mean": float(last.get("r_util_mean", float("nan"))),
        "r_cost_mean": float(last.get("r_cost_mean", float("nan"))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="results/single_agent/qwen_model_sweep")
    p.add_argument("-o", "--out", default="figures/fig_model_sweep.pdf")
    args = p.parse_args()

    root = Path(args.sweep_dir)
    if not root.exists():
        raise SystemExit(f"No sweep directory at {root}")

    # Collect: dict[tag] -> list[seed_metrics]
    per_model: Dict[str, List[dict]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        csvs = sorted(sub.glob("seed*.csv"))
        if not csvs:
            continue
        per_model[sub.name] = [_final_metrics(c) for c in csvs]

    if not per_model:
        raise SystemExit("No per-model CSVs found.")

    # Aggregate
    rows = []
    for tag, ms in per_model.items():
        rows.append({
            "tag": tag,
            "params_b": _param_count_b(tag),
            "gate_easy_mean": np.mean([m["gate_easy"] for m in ms]),
            "gate_easy_std":  np.std([m["gate_easy"] for m in ms]),
            "gate_hard_mean": np.mean([m["gate_hard"] for m in ms]),
            "gate_hard_std":  np.std([m["gate_hard"] for m in ms]),
            "R_mean":         np.mean([m["ep_return_10"] for m in ms]),
            "R_std":          np.std([m["ep_return_10"] for m in ms]),
            "n_seeds": len(ms),
        })
    df_agg = pd.DataFrame(rows).sort_values("params_b").reset_index(drop=True)
    print(df_agg.to_string())

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

    # (a) Easy vs hard gate% by LLM size
    ax = axes[0]
    x = np.arange(len(df_agg))
    width = 0.35
    ax.bar(x - width/2, 100*df_agg["gate_hard_mean"],
           yerr=100*df_agg["gate_hard_std"], width=width,
           color=COLORS["accent"], label=f"Hard gate{pct()}", capsize=4)
    ax.bar(x + width/2, 100*df_agg["gate_easy_mean"],
           yerr=100*df_agg["gate_easy_std"], width=width,
           color=COLORS["green"], label=f"Easy gate{pct()}", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['tag']}\n({r['params_b']:.0f}B)" for _, r in df_agg.iterrows()])
    ax.set_ylabel(f"Pr(invoke LLM) [{pct()}]")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Selective gating vs LLM size")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # (b) Reward vs LLM size
    ax = axes[1]
    ax.errorbar(df_agg["params_b"], df_agg["R_mean"],
                yerr=df_agg["R_std"], fmt="o-",
                color=COLORS["primary"], linewidth=2, markersize=8, capsize=4)
    for _, r in df_agg.iterrows():
        ax.annotate(r["tag"], (r["params_b"], r["R_mean"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color=COLORS["primary"])
    ax.set_xscale("log")
    ax.set_xlabel("LLM parameter count (B)")
    ax.set_ylabel("Episode return (100-step rolling)")
    ax.set_title("(b) Quality vs LLM size")
    ax.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out)


if __name__ == "__main__":
    main()
