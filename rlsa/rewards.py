"""Composite reward used to train RLSA (paper Eq. 1).

    R = lambda_make * R_make + lambda_util * R_util + lambda_cost * R_cost

where

    R_make = -(tau_end - tau_arr) / d_hat          normalized flow time
    R_util = mean(u) - eta * std(u)                utilization minus imbalance
    R_cost = -beta * g * (dt_llm / dt_budget)      LLM-invocation cost

The module is numpy-only and side-effect free so it can be unit-tested without
building an environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RewardWeights:
    """Weights of the composite reward."""

    lambda_make: float = 1.0
    lambda_util: float = 0.5
    lambda_cost: float = 1.0
    util_imbalance_eta: float = 0.2
    # beta: the operator knob that moves the policy along the cost/quality frontier.
    cost_beta: float = 1.0
    # Latency budget per LLM call. A call that takes exactly this long costs
    # ``lambda_cost * cost_beta`` reward units.
    cost_budget_seconds: float = 2.0

    def as_dict(self) -> dict:
        return {
            "lambda_make": self.lambda_make,
            "lambda_util": self.lambda_util,
            "lambda_cost": self.lambda_cost,
            "util_imbalance_eta": self.util_imbalance_eta,
            "cost_beta": self.cost_beta,
            "cost_budget_seconds": self.cost_budget_seconds,
        }


def makespan_reward(actual_seconds: float, estimated_seconds: float) -> float:
    """Negative normalized flow time; -1 means the job finished on estimate."""
    if estimated_seconds <= 0:
        return -1.0
    return -float(actual_seconds) / float(estimated_seconds)


def utilization_reward(utilization: np.ndarray, imbalance_eta: float = 0.2) -> float:
    """Mean utilization minus a penalty on cross-resource imbalance."""
    u = np.asarray(utilization, dtype=np.float64).ravel()
    if u.size == 0:
        return 0.0
    return float(u.mean() - imbalance_eta * u.std())


def llm_cost_penalty(
    gate: int,
    llm_latency_seconds: float,
    budget_seconds: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Zero when the surrogate is used; negative and latency-proportional otherwise."""
    if not gate:
        return 0.0
    if budget_seconds <= 0:
        budget_seconds = 1.0
    return -beta * float(llm_latency_seconds) / float(budget_seconds)


def composite_reward(
    actual_seconds: float,
    estimated_seconds: float,
    utilization: np.ndarray,
    gate: int,
    llm_latency_seconds: float,
    weights: Optional[RewardWeights] = None,
) -> dict:
    """Return the total composite reward and its per-term breakdown.

    The breakdown values (``r_make``, ``r_util``, ``r_cost``) are *unweighted*;
    ``total`` applies the ``lambda_*`` weights.
    """
    w = weights or RewardWeights()
    r_make = makespan_reward(actual_seconds, estimated_seconds)
    r_util = utilization_reward(utilization, imbalance_eta=w.util_imbalance_eta)
    r_cost = llm_cost_penalty(
        gate,
        llm_latency_seconds,
        budget_seconds=w.cost_budget_seconds,
        beta=w.cost_beta,
    )
    total = w.lambda_make * r_make + w.lambda_util * r_util + w.lambda_cost * r_cost
    return {
        "total": float(total),
        "r_make": float(r_make),
        "r_util": float(r_util),
        "r_cost": float(r_cost),
    }
