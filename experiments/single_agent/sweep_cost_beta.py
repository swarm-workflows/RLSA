"""Sweep the LLM-cost weight beta in the gating environment (Fig. 5).

For each beta and seed: train RLSA, then evaluate the thresholded policy on a
fresh environment and record gate rates and the quality reward. Output is one
CSV row per (beta, seed); plot with ``experiments.plots.plot_cost_sweep``.

    python -m experiments.single_agent.sweep_cost_beta \\
        --betas 0,0.25,0.5,1,2,4,8 --seeds 3 \\
        --out results/single_agent/cost_sweep/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from experiments.single_agent.common import evaluate, make_trainer
from rlsa.envs.gating_env import GatingEnvConfig
from rlsa.rewards import RewardWeights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--betas", default="0,0.25,0.5,1,2,4,8")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=12_000)
    p.add_argument("--pretrain", type=int, default=2_000)
    p.add_argument("--n-eval", type=int, default=2_000)
    p.add_argument("--out", default="results/single_agent/cost_sweep/sweep.csv")
    args = p.parse_args()

    betas = [float(x) for x in args.betas.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for beta in betas:
        env_cfg = GatingEnvConfig(reward_weights=RewardWeights(cost_beta=beta))
        for seed in range(args.seeds):
            t_run = time.time()
            trainer = make_trainer(env_cfg, seed)
            trainer.train(total_timesteps=args.timesteps, log_every=10**9,
                          pretrain_steps=args.pretrain)
            m = evaluate(trainer.policy, env_cfg, seed=1000 + seed, n_steps=args.n_eval)
            rows.append({"cost_beta": beta, "seed": seed, **m})
            print(f"beta={beta:5.2f} seed={seed}  quality_R={m['quality_R']:+.3f} "
                  f"gate_easy={100 * m['gate_easy']:5.1f}% gate_hard={100 * m['gate_hard']:5.1f}% "
                  f"({time.time() - t_run:.0f}s)")
            with open(out, "w", newline="") as f:  # rewrite after every run
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
