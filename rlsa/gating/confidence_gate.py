"""Non-RL confidence-gate baseline.

Reuses a trained RLSA policy's encoder and surrogate (frozen) and adds a small
residual-error head ``r(z) ~ |surrogate_bid - LLM_bid|``. The LLM is invoked when
the predicted error exceeds a threshold ``tau``. Sweeping ``tau`` lets the
baseline be compared to RLSA at a matched LLM budget and with identical
surrogate quality. It never looks at the ground-truth job difficulty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHead(nn.Module):
    """r(z) -> predicted absolute bid error (softplus, >= 0)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z)).squeeze(-1)


class ConfidenceGate:
    """Frozen RLSA surrogate + residual-error threshold. The RL gate head is unused."""

    def __init__(self, policy, residual_head: ResidualHead, tau: float):
        self.policy = policy
        self.residual = residual_head
        self.tau = float(tau)
        self.policy.eval()
        self.residual.eval()

    @torch.no_grad()
    def decide(self, obs_t: torch.Tensor):
        """Return ``(gate, surrogate_bid, predicted_error)`` for one observation.

        ``obs_t`` has shape ``(1, obs_dim)``.
        """
        z = self.policy.encoder(obs_t)
        sur_bid = float(self.policy.surrogate(z).reshape(-1)[0].item())
        err_hat = float(self.residual(z).reshape(-1)[0].item())
        gate = 1 if err_hat > self.tau else 0
        return gate, sur_bid, err_hat


def train_residual_head(
    policy,
    obs: torch.Tensor,
    err: torch.Tensor,
    epochs: int = 400,
    lr: float = 1e-3,
) -> tuple[ResidualHead, float]:
    """Fit a :class:`ResidualHead` on the frozen encoder's latents.

    ``obs`` holds observations from gated steps of a trained RLSA run and ``err``
    the matching ``|LLM_bid - surrogate_bid|``. Returns the head and final MSE.
    """
    policy.eval()
    with torch.no_grad():
        z = policy.encoder(obs)
    head = ResidualHead(hidden_dim=policy.cfg.hidden_dim)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        loss = ((head(z) - err) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head, float(loss.item())
