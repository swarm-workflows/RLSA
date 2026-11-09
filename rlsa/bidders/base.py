"""Bidder interface: what the gate=1 path calls.

A bidder scores an ``(AgentState, JobSpec)`` pair on a 0..100 scale and reports
how long it took. The reward penalizes that latency, so the gate learns when a
call is worth it. This module has no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class AgentState:
    """The part of an agent's observation that a bidder (or a human) would use."""

    cpu_util: float  # [0, 1]
    gpu_util: float  # [0, 1]
    ram_util: float  # [0, 1]
    disk_util: float  # [0, 1]
    queue_length: int
    neighbor_load: float  # mean neighbor utilization, [0, 1]
    historical_success_rate: float = 1.0  # [0, 1]


@dataclass
class JobSpec:
    """The job being bid on."""

    cpus: int
    gpus: int = 0
    ram_gb: float = 0.0
    disk_gb: float = 0.0
    estimated_seconds: float = 0.0
    user: str = ""
    job_id: str = ""


@dataclass
class BidResult:
    """One bid response. ``succeeded=False`` means a fallback score was returned."""

    score: float  # [0, 100]
    latency_seconds: float
    succeeded: bool = True
    explanation: str = ""
    provider: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)


class Bidder(Protocol):
    """Anything that can produce a bid.

    ``true_bid`` is only consumed by the mock oracle; real bidders ignore it.
    """

    def bid(
        self, agent_state: AgentState, job: JobSpec, true_bid: Optional[float] = None
    ) -> BidResult: ...

    @property
    def provider_name(self) -> str: ...


def capacity_bid(cpu_util: float, gpu_util: float, ram_util: float, disk_util: float) -> float:
    """Resource-based heuristic bid in [0, 100]: more free capacity, higher bid.

    Used by the ``capacity`` / ``always_surrogate`` baselines of the auction.
    """
    free_cpu, free_gpu, free_ram, free_disk = (
        1.0 - cpu_util,
        1.0 - gpu_util,
        1.0 - ram_util,
        1.0 - disk_util,
    )
    return float(100.0 * (0.4 * free_cpu + 0.1 * free_gpu + 0.3 * free_ram + 0.2 * free_disk))
