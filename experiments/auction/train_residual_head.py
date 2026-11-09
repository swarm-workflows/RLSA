"""Train the confidence-gate residual head for one RLSA checkpoint.

Inputs come from an RLSA auction run with ``--save-policy-path`` and
``--dump-residual-data``. The encoder of the RLSA policy is frozen; only the
residual-error head is fit.

    python -m experiments.auction.train_residual_head \\
        --data results/auction/headline_mock/policies/residdata_seed0.npz \\
        --policy results/auction/headline_mock/policies/rlsa_seed0.pt \\
        --out results/auction/headline_mock/policies/residual_seed0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rlsa.gating import GatingConfig, GatingPolicy, train_residual_head


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--obs-dim", type=int, default=20)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    d = np.load(args.data)
    obs = torch.tensor(d["obs"], dtype=torch.float32)
    err = torch.tensor(d["err"], dtype=torch.float32)
    if len(err) < 8:
        print(f"WARNING: only {len(err)} samples in {args.data}; the head will be poorly fit.")

    policy = GatingPolicy(GatingConfig(obs_dim=args.obs_dim))
    policy.load_state_dict(torch.load(args.policy, map_location="cpu", weights_only=True))
    head, mse = train_residual_head(policy, obs, err, epochs=args.epochs, lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.out)
    print(f"{len(err)} samples, final MSE={mse:.4f}, "
          f"err range=[{float(err.min()):.2f}, {float(err.max()):.2f}] -> {args.out}")


if __name__ == "__main__":
    main()
