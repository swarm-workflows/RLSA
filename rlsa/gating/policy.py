"""RLSA per-agent gating policy (paper Fig. 1).

A shared encoder ``phi`` maps the local observation to a latent ``z``, which
feeds three heads:

* gate head ``G(z)``       -> probability of invoking the LLM
* surrogate head ``pi(z)`` -> fast bid in ``[bid_min, bid_max]``
* value head ``V(z)``      -> PPO critic (training only)

The LLM itself is called outside this module, by the environment / agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class GatingConfig:
    obs_dim: int
    hidden_dim: int = 128
    bid_min: float = 0.0
    bid_max: float = 100.0
    gate_threshold: float = 0.5  # used by ``act(deterministic=True)``


class StateEncoder(nn.Module):
    """phi(o): two-layer MLP with LayerNorm."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SurrogateBidder(nn.Module):
    """pi(z): fast bid score in ``[bid_min, bid_max]``."""

    def __init__(self, hidden_dim: int, bid_min: float, bid_max: float):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)
        self.bid_min = bid_min
        self.bid_max = bid_max

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.head(z)).squeeze(-1)
        return self.bid_min + (self.bid_max - self.bid_min) * s


class GatingHead(nn.Module):
    """G(z): probability of invoking the LLM."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(z)).squeeze(-1)


class ValueHead(nn.Module):
    """V(z): state value for the PPO critic."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


class GatingPolicy(nn.Module):
    """Encoder + gate + surrogate + value head.

    ``forward`` returns ``(bid, p_gate, value, latent)``.
    """

    def __init__(self, cfg: GatingConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.surrogate = SurrogateBidder(cfg.hidden_dim, cfg.bid_min, cfg.bid_max)
        self.gate = GatingHead(cfg.hidden_dim)
        self.value = ValueHead(cfg.hidden_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        return self.surrogate(z), self.gate(z), self.value(z), z

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> dict:
        """Single-step inference.

        With ``deterministic=True`` the gate is thresholded at
        ``cfg.gate_threshold``; moving the threshold changes the LLM call rate
        of a trained policy without retraining.
        """
        bid, p_gate, v, z = self.forward(obs)
        if deterministic:
            gate = (p_gate > self.cfg.gate_threshold).long()
        else:
            gate = torch.bernoulli(p_gate).long()
        return {"gate": gate, "bid": bid, "p_gate": p_gate, "value": v, "latent": z}

    def evaluate(
        self, obs: torch.Tensor, gate_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probability of the gate actions, value, and gate entropy."""
        _, p_gate, v, _ = self.forward(obs)
        eps = 1e-8
        log_p = gate_actions * torch.log(p_gate + eps) + (1.0 - gate_actions) * torch.log(
            1.0 - p_gate + eps
        )
        entropy = -(
            p_gate * torch.log(p_gate + eps) + (1.0 - p_gate) * torch.log(1.0 - p_gate + eps)
        )
        return log_p, v, entropy
