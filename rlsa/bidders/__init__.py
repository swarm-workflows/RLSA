"""Bidders: what the gate=1 path calls.

``MockLLMBidder`` is a perfect oracle with reported latency (used for all mock
experiments). ``HFLocalBidder`` runs a local HuggingFace model and is imported
lazily because it needs the optional ``transformers`` dependency.
"""

from rlsa.bidders.base import AgentState, Bidder, BidResult, JobSpec, capacity_bid
from rlsa.bidders.mock import MockLLMBidder

__all__ = [
    "AgentState",
    "Bidder",
    "BidResult",
    "JobSpec",
    "MockLLMBidder",
    "capacity_bid",
]
