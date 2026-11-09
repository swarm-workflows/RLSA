import numpy as np
import pytest

from rlsa.rewards import (
    RewardWeights,
    composite_reward,
    llm_cost_penalty,
    makespan_reward,
    utilization_reward,
)


def test_makespan_reward():
    assert makespan_reward(100.0, 100.0) == pytest.approx(-1.0)
    assert makespan_reward(200.0, 100.0) == pytest.approx(-2.0)
    assert makespan_reward(10.0, 0.0) == -1.0


def test_utilization_reward():
    assert utilization_reward(np.ones(4)) == pytest.approx(1.0)
    u = np.array([1.0, 0.0, 1.0, 0.0])
    assert utilization_reward(u, imbalance_eta=0.2) == pytest.approx(0.5 - 0.2 * 0.5)
    assert utilization_reward(np.array([])) == 0.0


def test_llm_cost_penalty():
    assert llm_cost_penalty(0, 5.0, 1.0, 1.0) == 0.0
    assert llm_cost_penalty(1, 2.0, budget_seconds=2.0, beta=3.0) == pytest.approx(-3.0)


def test_composite_reward_weights():
    w = RewardWeights(lambda_make=1.0, lambda_util=0.5, lambda_cost=1.0, cost_beta=2.0,
                      cost_budget_seconds=2.0)
    u = np.array([0.7, 0.6, 0.8, 0.5])
    r = composite_reward(120.0, 100.0, u, gate=1, llm_latency_seconds=1.0, weights=w)
    assert r["r_make"] == pytest.approx(-1.2)
    assert r["r_cost"] == pytest.approx(-1.0)
    assert r["total"] == pytest.approx(r["r_make"] + 0.5 * r["r_util"] + r["r_cost"])


def test_surrogate_is_free():
    u = np.full(4, 0.5)
    r = composite_reward(60.0, 60.0, u, gate=0, llm_latency_seconds=10.0)
    assert r["r_cost"] == 0.0
