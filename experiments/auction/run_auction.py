"""Run the threaded asynchronous auction for a grid of (algo, N, seed) cells.

Every cell writes ``<out-dir>/<prefix><algo>_N<n>_seed<s>.json`` with the full
metrics dict of :func:`rlsa.envs.distributed.run_distributed`.

Algorithms: random, capacity, always_surrogate, always_llm, oracle, rlsa,
confidence_gate (needs ``--load-policy-path`` and ``--residual-path``).

Path options accept ``{seed}``, ``{algo}`` and ``{n}`` placeholders.

Example (headline mock cell, CPU only)::

    python -m experiments.auction.run_auction \\
        --algos random,capacity,always_surrogate,always_llm,oracle,rlsa \\
        --ns 4 --seeds 10 --n-jobs 400 --pretrain-steps 80 \\
        --save-policy-path results/auction/headline_mock/policies/rlsa_seed{seed}.pt \\
        --dump-residual-data results/auction/headline_mock/policies/residdata_seed{seed}.npz \\
        --out-dir results/auction/headline_mock

See ``experiments/reproduce/`` for the exact command behind every paper result.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from rlsa.envs.auction import AuctionConfig
from rlsa.envs.distributed import DistributedConfig, run_distributed
from rlsa.rewards import RewardWeights

DEFAULT_ALGOS = "random,capacity,always_surrogate,always_llm,oracle,rlsa"


def _floats(s: str):
    return [float(x) for x in s.split(",")] if s.strip() else None


def _strs(s: str):
    return [x.strip() for x in s.split(",")] if s.strip() else None


def _per_agent(values, n):
    """Per-agent lists only apply when their length matches N."""
    return values if values and len(values) == n else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", default=DEFAULT_ALGOS)
    p.add_argument("--ns", default="4", help="comma-separated agent counts")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=400)
    p.add_argument("--prefix", default="", help="filename prefix, e.g. 'scale_'")
    p.add_argument("--out-dir", required=True)
    # bidders
    p.add_argument("--bidder", choices=["mock", "hf"], default="mock")
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--bidder-kinds", default="", help="per-agent mock|hf, comma-separated")
    p.add_argument("--hf-models-per-agent", default="", help="per-agent HF model ids")
    # network
    p.add_argument("--deadline-ms", type=float, default=50.0)
    p.add_argument("--net-mean-ms", type=float, default=2.0)
    p.add_argument("--net-std-ms", type=float, default=0.5)
    p.add_argument("--net-drop", type=float, default=0.0)
    p.add_argument("--net-mean-ms-per-agent", default="")
    p.add_argument("--net-std-ms-per-agent", default="")
    p.add_argument("--net-drop-per-agent", default="")
    p.add_argument("--tiebreak", choices=["deterministic", "random"], default="deterministic")
    # training
    p.add_argument("--sharing", choices=["ctde", "dtde"], default="ctde")
    p.add_argument("--pretrain-steps", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.0, help="0 = bandit advantage")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--cost-beta", type=float, default=None)
    p.add_argument("--save-policy-path", default="")
    p.add_argument("--load-policy-path", default="", help="evaluate this policy frozen")
    # confidence gate
    p.add_argument("--confidence-tau", type=float, default=0.0)
    p.add_argument("--residual-path", default="")
    p.add_argument("--dump-residual-data", default="")
    args = p.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    ns = [int(n) for n in args.ns.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = RewardWeights() if args.cost_beta is None else RewardWeights(cost_beta=args.cost_beta)

    total, idx = len(algos) * len(ns) * len(seeds), 0
    for algo in algos:
        for n in ns:
            for s in seeds:
                idx += 1
                fmt = dict(seed=s, algo=algo, n=n)
                cfg = DistributedConfig(
                    base=AuctionConfig(n_agents=n, n_jobs=args.n_jobs, reward_weights=weights),
                    bid_deadline_ms=args.deadline_ms,
                    net_mean_ms=args.net_mean_ms,
                    net_std_ms=args.net_std_ms,
                    net_drop_prob=args.net_drop,
                    net_mean_ms_per_agent=_per_agent(_floats(args.net_mean_ms_per_agent), n),
                    net_std_ms_per_agent=_per_agent(_floats(args.net_std_ms_per_agent), n),
                    net_drop_prob_per_agent=_per_agent(_floats(args.net_drop_per_agent), n),
                    bidder_kinds=_per_agent(_strs(args.bidder_kinds), n),
                    bidder_kind=args.bidder,
                    hf_model_name=args.hf_model,
                    hf_model_names_per_agent=_per_agent(_strs(args.hf_models_per_agent), n),
                    sharing_mode=args.sharing,
                    pretrain_steps=args.pretrain_steps,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    tiebreak=args.tiebreak,
                    confidence_tau=args.confidence_tau,
                    residual_path=args.residual_path.format(**fmt),
                    dump_residual_data=args.dump_residual_data.format(**fmt),
                    save_policy_path=args.save_policy_path.format(**fmt),
                    load_policy_path=args.load_policy_path.format(**fmt),
                )
                out_path = out_dir / f"{args.prefix}{algo}_N{n}_seed{s}.json"
                print(f"[{idx}/{total}] algo={algo} N={n} seed={s}", flush=True)
                t0 = time.time()
                try:
                    result = run_distributed(algo, cfg, seed=s)
                except Exception as e:  # keep the sweep going, but say so loudly
                    traceback.print_exc()
                    print(f"  FAILED {algo} N={n} seed={s}: {e}", flush=True)
                    continue
                result["algo"] = algo
                result["seed"] = s
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  makespan={result['makespan']:.0f} llm/job={result['llm_calls_per_job']:.2f} "
                      f"jain={result['jain_fairness']:.3f} -> {out_path} ({time.time() - t0:.1f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
