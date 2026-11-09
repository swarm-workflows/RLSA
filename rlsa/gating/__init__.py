"""Gating policy network and the non-RL confidence-gate baseline."""

from rlsa.gating.confidence_gate import ConfidenceGate, ResidualHead, train_residual_head
from rlsa.gating.policy import GatingConfig, GatingPolicy

__all__ = [
    "ConfidenceGate",
    "GatingConfig",
    "GatingPolicy",
    "ResidualHead",
    "train_residual_head",
]
