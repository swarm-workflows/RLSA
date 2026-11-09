"""Shared helpers for the single-agent gating experiments."""

from __future__ import annotations

import torch

from rlsa.envs.gating_env import GatingEnv, GatingEnvConfig
from rlsa.ppo import PPOConfig, RLSATrainer


def make_trainer(
    env_cfg: GatingEnvConfig,
    seed: int,
    rollout: int = 512,
    gamma: float = 0.0,
    gae_lambda: float = 0.0,
    imitation_coef: float = 1.0,
) -> RLSATrainer:
    """PPO trainer with the paper's defaults (bandit advantage, gamma = 0)."""
    return RLSATrainer(
        env_fn=lambda: GatingEnv(cfg=env_cfg, seed=seed),
        cfg=PPOConfig(
            obs_dim=env_cfg.obs_dim,
            rollout_length=rollout,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=0.02,
            value_coef=0.5,
            surrogate_imitation_coef=imitation_coef,
        ),
    )


def evaluate(policy, env_cfg: GatingEnvConfig, seed: int, n_steps: int) -> dict:
    """Run the deterministic (thresholded) policy and report per-bucket stats.

    ``quality_R`` strips the (weighted) LLM-cost term so that policies trained
    with different cost weights can be compared on one axis.
    """
    env = GatingEnv(cfg=env_cfg, seed=seed)
    stats = {b: {"n": 0, "gate": 0, "reward": 0.0, "err": 0.0} for b in ("easy", "hard")}
    quality = cost = 0.0
    obs, _ = env.reset()
    for _ in range(n_steps):
        out = policy.act(torch.as_tensor(obs).float().unsqueeze(0), deterministic=True)
        action = {"bid": float(out["bid"].item()), "gate": int(out["gate"].item())}
        obs, r, term, trunc, info = env.step(action)
        s = stats["hard" if info["is_hard_job"] else "easy"]
        s["n"] += 1
        s["gate"] += action["gate"]
        s["reward"] += float(r)
        s["err"] += float(info["score_error"])
        c = float(info["reward_breakdown"]["r_cost"]) * env_cfg.reward_weights.lambda_cost
        cost += c
        quality += float(r) - c
        if term or trunc:
            obs, _ = env.reset()

    e, h = stats["easy"], stats["hard"]
    n = max(1, e["n"] + h["n"])
    return {
        "overall_R": (e["reward"] + h["reward"]) / n,
        "quality_R": quality / n,
        "cost_R": cost / n,
        "gate_overall": (e["gate"] + h["gate"]) / n,
        "gate_easy": e["gate"] / max(1, e["n"]),
        "gate_hard": h["gate"] / max(1, h["n"]),
        "easy_R": e["reward"] / max(1, e["n"]),
        "hard_R": h["reward"] / max(1, h["n"]),
        "score_err_easy": e["err"] / max(1, e["n"]),
        "score_err_hard": h["err"] / max(1, h["n"]),
    }
