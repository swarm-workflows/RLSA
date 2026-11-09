import csv

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rlsa.bidders import MockLLMBidder  # noqa: E402
from rlsa.envs.gating_env import GatingEnv, GatingEnvConfig  # noqa: E402
from rlsa.ppo import PPOConfig, RLSATrainer  # noqa: E402


def test_spaces_and_step():
    env = GatingEnv(seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step({"bid": 50.0, "gate": 0})
    assert env.observation_space.contains(obs)
    assert np.isfinite(r) and not term
    for k in ("is_hard_job", "true_bid", "effective_bid", "score_error", "reward_breakdown"):
        assert k in info


def test_env_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(GatingEnv(seed=0), skip_render_check=True)


def test_seed_determinism():
    def rollout(seed):
        env = GatingEnv(seed=seed)
        env.reset()
        return [env.step({"bid": 50.0, "gate": i % 2})[1] for i in range(20)]

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_oracle_gate_is_exact_and_costs_latency():
    env = GatingEnv(seed=0)
    env.reset()
    _, _, _, _, info = env.step({"bid": 0.0, "gate": 1})
    assert info["score_error"] == 0.0
    assert info["reward_breakdown"]["r_cost"] < 0


def test_episode_truncates():
    env = GatingEnv(GatingEnvConfig(max_steps_per_episode=5), seed=0)
    env.reset()
    flags = [env.step({"bid": 50.0, "gate": 0})[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]


def test_external_bidder_is_used():
    env = GatingEnv(GatingEnvConfig(bidder=MockLLMBidder(latency_mean=0.3, latency_std=0.0, seed=0)), seed=0)
    env.reset()
    _, _, _, _, info = env.step({"bid": 0.0, "gate": 1})
    assert info["llm_latency"] == pytest.approx(0.3)
    assert info["score_error"] == pytest.approx(0.0, abs=1e-6)


def test_ppo_smoke(tmp_path):
    cfg = GatingEnvConfig()
    trainer = RLSATrainer(
        env_fn=lambda: GatingEnv(cfg=cfg, seed=0),
        cfg=PPOConfig(obs_dim=cfg.obs_dim, rollout_length=128, gamma=0.0, gae_lambda=0.0),
    )
    out = tmp_path / "seed0.csv"
    trainer.train(total_timesteps=512, log_every=100, pretrain_steps=128, csv_log_path=str(out))
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4
    assert {"gate_easy", "gate_hard", "r_make_mean", "r_cost_mean"} <= set(rows[0])
