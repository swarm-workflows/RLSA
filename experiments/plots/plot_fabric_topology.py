"""Geographic layout of the FABRIC slice (paper figure ``fig_fabric_topology``).

Draws US state boundaries from the bundled ``data/us-states.geojson`` and the
three FABRIC sites used in the paper (Starlight/Chicago, Michigan,
Massachusetts), sized by VM count, with the measured inter-site RTTs.

    python -m experiments.plots.plot_fabric_topology --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from experiments.common import REPO_ROOT
from experiments.plots.style import apply as apply_style
from experiments.plots.style import method_color, save

C_COORD = method_color("rlsa")
C_AGENT = method_color("always_llm")
GEO = REPO_ROOT / "data" / "us-states.geojson"

# Sites: real lon/lat, host state, role, VM counts, and label placement.
SITES = {
    "STAR": dict(lon=-87.6, lat=41.9, state="Illinois", name="Starlight (Chicago)",
                 agents=5, coord=False, lbldx=0, lbldy=-60, ha="center"),
    "MICH": dict(lon=-84.5, lat=43.3, state="Michigan", name="Michigan",
                 agents=5, coord=False, lbldx=-8, lbldy=48, ha="center"),
    "MASS": dict(lon=-71.8, lat=42.3, state="Massachusetts", name="Massachusetts",
                 agents=6, coord=True, lbldx=14, lbldy=40, ha="center"),
}
HOST_STATES = {s["state"] for s in SITES.values()}

# Context states to outline (region around the sites) so the map reads clearly.
REGION_STATES = {
    "Illinois", "Michigan", "Massachusetts", "Indiana", "Ohio", "Wisconsin",
    "New York", "Pennsylvania", "Iowa", "Missouri", "Kentucky", "West Virginia",
    "Virginia", "Maryland", "New Jersey", "Connecticut", "Rhode Island",
    "Vermont", "New Hampshire", "Maine", "Minnesota", "Tennessee", "Delaware",
}


def _polys(geom):
    """Yield each polygon's exterior ring as (xs, ys) from a GeoJSON geometry."""
    t, coords = geom["type"], geom["coordinates"]
    rings = coords if t == "Polygon" else [p for poly in coords for p in poly]
    for ring in rings:
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        yield xs, ys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    with open(GEO) as f:
        geo = json.load(f)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    # Draw state boundaries: host states shaded, context states light outline.
    for feat in geo["features"]:
        name = feat["properties"].get("name")
        if name not in REGION_STATES:
            continue
        host = name in HOST_STATES
        fc = "#dfe8f3" if host else "#f5f5f5"
        ec = "#9bb3d4" if host else "#cccccc"
        lw = 1.1 if host else 0.7
        for xs, ys in _polys(feat["geometry"]):
            ax.fill(xs, ys, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)

    # WAN links + measured RTT labels (offsets keep labels off the nodes/links).
    links = [("STAR", "MICH", "~5 ms", 0.0,  (0.3, 1.3)),
             ("MICH", "MASS", "~24 ms", -0.34, (0.0, 3.0)),
             ("STAR", "MASS", "~27 ms", 0.34, (0.0, -3.2))]
    for a, b, rtt, rad, (ldx, ldy) in links:
        A, B = SITES[a], SITES[b]
        ax.add_patch(FancyArrowPatch(
            (A["lon"], A["lat"]), (B["lon"], B["lat"]),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color="#5a5a5a", lw=1.4, linestyle=(0, (5, 3)), zorder=3))
        mx, my = (A["lon"] + B["lon"]) / 2, (A["lat"] + B["lat"]) / 2
        ax.annotate(rtt, (mx + ldx, my + ldy), ha="center", va="center",
                    fontsize=8.4, color="#2a2a2a", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92))

    # Site nodes (scatter = circular regardless of aspect) + labels.
    for key, s in SITES.items():
        col = C_COORD if s["coord"] else C_AGENT
        vms = s["agents"] + (1 if s["coord"] else 0)
        size = (15 + 2.3 * vms) ** 2
        ax.scatter([s["lon"]], [s["lat"]], s=size, c=col, edgecolors="white",
                   linewidths=1.6, alpha=0.97, zorder=5)
        ax.annotate(str(vms), (s["lon"], s["lat"]), ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="white", zorder=7)
        role = "coordinator + " if s["coord"] else ""
        label = f"{key}\n{s['name']}\n{role}{s['agents']} agents"
        ax.annotate(label, (s["lon"], s["lat"]), textcoords="offset points",
                    xytext=(s["lbldx"], s["lbldy"]), ha=s["ha"], va="center",
                    fontsize=8.6, color="#1f1f1f", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.1, alpha=0.97))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_COORD,
               markersize=12, label="Coordinator site"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AGENT,
               markersize=12, label="Agent site"),
        Line2D([0], [0], color="#5a5a5a", linestyle=(0, (5, 3)),
               label="FABNet IPv4 WAN link (real RTT)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.2, framealpha=0.94)

    ax.set_title("FABRIC real-WAN slice: 17 VMs across 3 sites "
                 "(1 coordinator + 16 agents)", fontsize=10.5)
    ax.set_xlim(-93.5, -68.5)
    ax.set_ylim(36.5, 47.8)
    ax.set_aspect(1.30)  # mild correction for mid-US latitudes
    # Schematic map: drop the lon/lat axis decoration entirely.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = out_dir / "fig_fabric_topology.pdf"
    save(fig, out)


if __name__ == "__main__":
    main()
