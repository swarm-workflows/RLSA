"""Paired significance tests between algorithms across shared seeds.

Every seed fixes the job stream for all algorithms, so a seed is a valid unit of
paired replication. Reports a paired t-test and the Wilcoxon signed-rank test.

    python -m experiments.auction.paired_stats --run-dir results/auction/headline_mock \\
        --ref rlsa --others always_llm

Cells with different filename prefixes (e.g. the confidence gate at tau=10) can
be compared with ``--other-dir`` / ``--other-prefix``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

METRICS = ["makespan", "llm_calls_per_job", "jain_fairness"]


def load_by_seed(run_dir, algo, n, prefix=""):
    out = {}
    for f in glob.glob(str(Path(run_dir) / f"{prefix}{algo}_N{n}_seed*.json")):
        d = json.load(open(f))
        out[int(d.get("seed", Path(f).stem.split("seed")[-1]))] = d
    return out


def paired(ref: dict, other: dict, metric: str) -> dict:
    seeds = sorted(set(ref) & set(other))
    a = np.array([ref[s][metric] for s in seeds])
    b = np.array([other[s][metric] for s in seeds])
    res = {"n_pairs": len(seeds), "ref_mean": float(a.mean()), "other_mean": float(b.mean()),
           "mean_diff": float((a - b).mean()), "t_p": float("nan"), "wilcoxon_p": float("nan")}
    if _st is not None and len(seeds) >= 2 and np.std(a - b) > 0:
        res["t_p"] = float(_st.ttest_rel(a, b).pvalue)
        try:
            res["wilcoxon_p"] = float(_st.wilcoxon(a, b).pvalue)
        except Exception:
            pass
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--ref", default="rlsa")
    p.add_argument("--prefix", default="")
    p.add_argument("--others", default="always_llm")
    p.add_argument("--other-dir", default="", help="defaults to --run-dir")
    p.add_argument("--other-prefix", default=None, help="defaults to --prefix")
    args = p.parse_args()

    if _st is None:
        print("WARNING: scipy not installed; reporting means only (pip install -e '.[plot]').")
    ref = load_by_seed(args.run_dir, args.ref, args.n, args.prefix)
    if not ref:
        raise SystemExit(f"no cells for ref={args.ref} in {args.run_dir}")
    other_dir = args.other_dir or args.run_dir
    other_prefix = args.prefix if args.other_prefix is None else args.other_prefix
    for algo in [a for a in args.others.split(",") if a.strip()]:
        other = load_by_seed(other_dir, algo, args.n, other_prefix)
        if not other:
            print(f"(skip {algo}: no cells)")
            continue
        print(f"\n=== {args.ref} vs {other_prefix}{algo} (paired over seeds) ===")
        for m in METRICS:
            r = paired(ref, other, m)
            print(f"  {m:<18} {r['ref_mean']:10.3f} vs {r['other_mean']:10.3f}  "
                  f"diff={r['mean_diff']:+.3f}  n={r['n_pairs']}  "
                  f"t-test p={r['t_p']:.3g}  wilcoxon p={r['wilcoxon_p']:.3g}")


if __name__ == "__main__":
    main()
