"""Mock LLM bidder: a perfect oracle with Gaussian (reported) latency.

The environment knows the ground-truth bid and passes it via ``true_bid``; the
mock echoes it back. Latency is *reported*, not slept, so training stays fast
while the reward still pays for every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rlsa.bidders.base import AgentState, BidResult, JobSpec


@dataclass
class MockLLMBidder:
    latency_mean: float = 2.0
    latency_std: float = 0.5
    default_true_bid: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @property
    def provider_name(self) -> str:
        return "mock"

    def bid(
        self,
        agent_state: AgentState,
        job: JobSpec,
        true_bid: Optional[float] = None,
    ) -> BidResult:
        score = true_bid if true_bid is not None else self.default_true_bid
        if score is None:
            # No ground truth supplied: fall back to a state-aware heuristic.
            avg_util = (
                agent_state.cpu_util
                + agent_state.gpu_util
                + agent_state.ram_util
                + agent_state.disk_util
            ) / 4.0
            score = float(np.clip(100.0 * (1.0 - avg_util), 0.0, 100.0))
        latency = float(max(0.01, self._rng.normal(self.latency_mean, self.latency_std)))
        return BidResult(
            score=float(np.clip(score, 0.0, 100.0)),
            latency_seconds=latency,
            succeeded=True,
            explanation="mock oracle",
            provider=self.provider_name,
            model="mock",
        )
