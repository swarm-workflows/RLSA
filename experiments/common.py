"""Small helpers shared by the experiment and plotting scripts."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"


def load_json_glob(pattern: str | Path) -> list[dict]:
    """Load every JSON file matching ``pattern`` (sorted by path)."""
    out = []
    for f in sorted(glob.glob(str(pattern))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def bootstrap_ci(xs, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI."""
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return 0.0, float("nan"), float("nan")
    if len(xs) < 2:
        return float(xs.mean()), float(xs[0]), float(xs[0])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(xs, size=len(xs), replace=True).mean() for _ in range(n_boot)])
    return float(xs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
