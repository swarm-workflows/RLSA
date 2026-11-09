"""Train RLSA in the controlled gating environment (Figs. 2-4).

Writes one CSV per seed with per-PPO-update metrics (gate rate on easy / hard
jobs, bid error, reward components, losses); plot them with
``experiments.plots.plot_training``.

Mock oracle bidder (Fig. 2, CPU, a few minutes per seed)::

    python -m experiments.single_agent.train --seeds 5 \\
        --out-dir results/single_agent/training

Real Qwen-2.5 bidders (Figs. 3-4, one GPU; needs ``pip install -e ".[hf]"``)::

    python -m experiments.single_agent.train --bidder hf \\
        --models Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct \\
        --seeds 3 --timesteps 4000 --rollout 256 --pretrain 1000 --prompt oneshot \\
        --out-dir results/single_agent/qwen_model_sweep

With ``--bidder hf`` the CSVs go to ``<out-dir>/<model-tag>/seed<k>.csv``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from experiments.single_agent.common import make_trainer
from rlsa.envs.gating_env import GatingEnvConfig
from rlsa.rewards import RewardWeights


def model_tag(model_name: str) -> str:
    """'Qwen/Qwen2.5-7B-Instruct' -> 'qwen2.5-7b'."""
    return model_name.lower().split("/")[-1].replace("-instruct", "").strip("-")


def train_seeds(env_cfg, seeds, out_dir: Path, timesteps, rollout, pretrain, log_every):
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        print(f"=== seed {seed} -> {out_dir}/seed{seed}.csv")
        trainer = make_trainer(env_cfg, seed, rollout=rollout)
        trainer.train(
            total_timesteps=timesteps,
            log_every=log_every,
            pretrain_steps=pretrain,
            csv_log_path=str(out_dir / f"seed{seed}.csv"),
        )
        print(f"    seed {seed} done in {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bidder", choices=["mock", "hf"], default="mock")
    p.add_argument("--models", default="Qwen/Qwen2.5-3B-Instruct",
                   help="comma-separated HF model ids (with --bidder hf)")
    p.add_argument("--prompt", default="oneshot",
                   help="HF prompt template: baseline | explicit | cot_lite | oneshot")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--rollout", type=int, default=512)
    p.add_argument("--pretrain", type=int, default=3_000)
    p.add_argument("--cost-beta", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out-dir", default="results/single_agent/training")
    args = p.parse_args()

    weights = RewardWeights(cost_beta=args.cost_beta)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    out_dir = Path(args.out_dir)

    if args.bidder == "mock":
        train_seeds(GatingEnvConfig(reward_weights=weights), seeds, out_dir,
                    args.timesteps, args.rollout, args.pretrain, args.log_every)
        return

    from rlsa.bidders.hf_local import HFLocalBidder

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        bidder = HFLocalBidder(model_name=model, dtype=args.dtype, prompt_name=args.prompt)
        env_cfg = GatingEnvConfig(reward_weights=weights, bidder=bidder)
        train_seeds(env_cfg, seeds, out_dir / model_tag(model),
                    args.timesteps, args.rollout, args.pretrain, args.log_every)
        del bidder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
