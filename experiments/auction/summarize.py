"""Summarize auction result cells: mean and 95% bootstrap CI per algorithm.

Reads ``<run-dir>/<prefix><algo>_N<n>_seed*.json``. This is the aggregation
behind every auction number quoted in the paper.

    python -m experiments.auction.summarize --run-dir results/auction/headline_mock
    python -m experiments.auction.summarize --run-dir results/auction/confidence_gate \\
        --algos confidence_gate --prefix tau10_
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from experiments.common import bootstrap_ci

ALL_ALGOS = "random,capacity,always_surrogate,always_llm,oracle,rlsa,confidence_gate"


def collect(run_dir, algo, n, prefix=""):
    files = sorted(glob.glob(str(Path(run_dir) / f"{prefix}{algo}_N{n}_seed*.json")))
    rows = [json.load(open(f)) for f in files]
    return (
        [r["makespan"] for r in rows],
        [r["llm_calls_per_job"] for r in rows],
        [r["jain_fairness"] for r in rows],
    )


def summarize(run_dir, n=4, algos=ALL_ALGOS, prefix="", n_boot=10_000) -> list[dict]:
    rows = []
    for algo in [a for a in algos.split(",") if a.strip()]:
        mksp, llmj, jain = collect(run_dir, algo, n, prefix)
        if not mksp:
            continue
        m, c, j = (bootstrap_ci(x, n_boot=n_boot) for x in (mksp, llmj, jain))
        rows.append({
            "algo": algo, "seeds": len(mksp),
            "makespan_mean": round(m[0], 1), "makespan_lo": round(m[1], 1), "makespan_hi": round(m[2], 1),
            "llm_per_job_mean": round(c[0], 3), "llm_per_job_lo": round(c[1], 3), "llm_per_job_hi": round(c[2], 3),
            "jain_mean": round(j[0], 3), "jain_lo": round(j[1], 3), "jain_hi": round(j[2], 3),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--algos", default=ALL_ALGOS)
    p.add_argument("--prefix", default="")
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    rows = summarize(args.run_dir, args.n, args.algos, args.prefix)
    if not rows:
        print(f"no cells matching {args.prefix}<algo>_N{args.n}_seed*.json in {args.run_dir}")
        return
    print(f"{'algo':<18} {'k':>3}  {'makespan [95% CI]':>26}  {'LLM/job [95% CI]':>22}  {'Jain [95% CI]':>22}")
    for r in rows:
        print(f"{r['algo']:<18} {r['seeds']:>3}  "
              f"{r['makespan_mean']:8.0f} [{r['makespan_lo']:7.0f},{r['makespan_hi']:7.0f}]  "
              f"{r['llm_per_job_mean']:6.2f} [{r['llm_per_job_lo']:5.2f},{r['llm_per_job_hi']:5.2f}]  "
              f"{r['jain_mean']:6.3f} [{r['jain_lo']:5.3f},{r['jain_hi']:5.3f}]")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
