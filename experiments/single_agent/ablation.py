"""Ablation of RLSA's design choices in the gating environment (Table II).

Arms:
    full         imitation pretraining + auxiliary teacher loss + bandit advantage (gamma=0)
    no_pretrain  no imitation pretraining
    no_aux       no auxiliary teacher loss
    discounted   standard discounted GAE (gamma=0.99, lambda=0.95)

    python -m experiments.single_agent.ablation --seeds 3 \\
        --out results/single_agent/ablation/ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.single_agent.common import evaluate, make_trainer
from rlsa.envs.gating_env import GatingEnvConfig
from rlsa.rewards import RewardWeights

ARMS = {
    "full": {"pretrain": True, "aux_coef": 1.0, "gamma": 0.0, "gae_lambda": 0.0},
    "no_pretrain": {"pretrain": False, "aux_coef": 1.0, "gamma": 0.0, "gae_lambda": 0.0},
    "no_aux": {"pretrain": True, "aux_coef": 0.0, "gamma": 0.0, "gae_lambda": 0.0},
    "discounted": {"pretrain": True, "aux_coef": 1.0, "gamma": 0.99, "gae_lambda": 0.95},
}

ARM_LABELS = {
    "full": "Full RLSA",
    "no_pretrain": "- imitation pretraining",
    "no_aux": "- auxiliary teacher loss",
    "discounted": "- bandit GAE (gamma=0.99)",
}


def print_table(rows: list[dict]) -> None:
    print(f"\n{'Variant':<28} {'Overall R':>16} {'Easy gate%':>14} {'Hard gate%':>14}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        R = np.array([float(r["overall_R"]) for r in sub])
        e = 100 * np.array([float(r["gate_easy"]) for r in sub])
        h = 100 * np.array([float(r["gate_hard"]) for r in sub])
        sd = (lambda x: x.std(ddof=1) if len(x) > 1 else 0.0)  # pandas-style sample std
        print(f"{ARM_LABELS[arm]:<28} {R.mean():+8.3f} ± {sd(R):.3f} "
              f"{e.mean():6.1f} ± {sd(e):4.1f} {h.mean():6.1f} ± {sd(h):4.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=12_000)
    p.add_argument("--pretrain", type=int, default=2_000)
    p.add_argument("--n-eval", type=int, default=2_000)
    p.add_argument("--out", default="results/single_agent/ablation/ablation.csv")
    p.add_argument("--summarize", action="store_true",
                   help="only print the table for an existing --out CSV")
    args = p.parse_args()

    out = Path(args.out)
    if args.summarize:
        with open(out) as f:
            print_table(list(csv.DictReader(f)))
        return

    env_cfg = GatingEnvConfig(reward_weights=RewardWeights())
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for arm, a in ARMS.items():
        for seed in range(args.seeds):
            trainer = make_trainer(env_cfg, seed, gamma=a["gamma"], gae_lambda=a["gae_lambda"],
                                   imitation_coef=a["aux_coef"])
            trainer.train(total_timesteps=args.timesteps, log_every=10**9,
                          pretrain_steps=args.pretrain if a["pretrain"] else 0)
            m = evaluate(trainer.policy, env_cfg, seed=1000 + seed, n_steps=args.n_eval)
            rows.append({"arm": arm, "seed": seed, **m})
            print(f"arm={arm:<12} seed={seed}  R={m['overall_R']:+.3f} "
                  f"gate_easy={100 * m['gate_easy']:5.1f}% gate_hard={100 * m['gate_hard']:5.1f}%")
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    print_table(rows)


if __name__ == "__main__":
    main()
