"""Shared plotting style for every paper figure.

One palette across the paper, aligned with the architecture diagram:
blue = RLSA / gate, red = LLM path, green = surrogate, amber = oracle, gray = baselines.

LaTeX text rendering is used when a ``latex`` binary is available (as for the
camera-ready figures); otherwise matplotlib's mathtext is used. Force either with
``RLSA_USETEX=1`` / ``RLSA_USETEX=0``.
"""

from __future__ import annotations

import os
import shutil

import matplotlib as mpl

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False

COLORS = {
    "primary": "#3b6ea5",  # RLSA / gate
    "accent": "#c44e52",  # Always-LLM / LLM path / hard jobs
    "green": "#55a868",  # surrogate / easy jobs
    "amber": "#dd8452",  # oracle
    "purple": "#8172b3",  # confidence gate
    "muted": "#9aa5b1",  # random / capacity
    "dark": "#3a3f4b",
    "ref": "#b0473a",  # reference lines
    "grid": "#dfe3e8",
}

CYCLE = [COLORS[k] for k in ("primary", "accent", "green", "amber", "purple", "muted")]

METHOD = {
    "rlsa": COLORS["primary"],
    "always_llm": COLORS["accent"],
    "oracle": COLORS["amber"],
    "always_surrogate": COLORS["green"],
    "capacity": COLORS["muted"],
    "random": COLORS["muted"],
    "confidence_gate": COLORS["purple"],
    "hard": COLORS["accent"],
    "easy": COLORS["green"],
}

LABELS = {
    "random": "Rand",
    "capacity": "Cap",
    "always_surrogate": "Surr",
    "always_llm": "Always-LLM",
    "oracle": "Oracle",
    "rlsa": "RLSA",
    "confidence_gate": "Conf-gate",
}


def method_color(name: str, default: str | None = None) -> str:
    return METHOD.get(str(name).strip().lower(), default or COLORS["muted"])


def label(name: str) -> str:
    return LABELS.get(str(name).strip().lower(), str(name))


def pct() -> str:
    """A percent sign that survives LaTeX rendering (call after ``apply()``)."""
    return r"\%" if mpl.rcParams["text.usetex"] else "%"


def save(fig, path) -> None:
    """Save ``path`` (PDF) plus a PNG sibling used by the documentation site."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    print("wrote", path)


def _use_tex() -> bool:
    env = os.environ.get("RLSA_USETEX")
    if env is not None:
        return env not in ("0", "false", "False", "")
    return shutil.which("latex") is not None


def apply() -> None:
    """Apply the global style. Call once before plotting."""
    if _HAS_SNS:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font="serif",
            rc={"axes.prop_cycle": mpl.cycler(color=CYCLE)},
        )
    mpl.rcParams.update({
        "text.usetex": _use_tex(),
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
        "axes.edgecolor": COLORS["dark"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "patch.linewidth": 0.0,
        "patch.edgecolor": "none",
        "patch.force_edgecolor": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })
