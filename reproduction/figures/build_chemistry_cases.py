"""Cell-native attributions on molecular case studies (appendix figure).

Builds figures/out/chemistry_case_studies.pdf: one row of four molecule
panels, each drawn as a 2D graph with atoms (rank 0), bonds (rank 1), and
ring cells (rank 2) colored by their Shapley attribution on a shared,
symmetric diverging scale. Ground-truth motif atoms carry a bold outline.

Input (read-only, frozen record):
    data/frozen/reference/synthesis_report/results/molecules_data.json

Style: repo rules — linear axes only (the panels are geometric, no data
axes), serif fonts, plain-text dataset names (no codenames), ~\\linewidth
width, vector PDF. Deterministic: fixed layout seed, no wall-clock input
(SOURCE_DATE_EPOCH pinned so the PDF is byte-stable across rebuilds).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF metadata

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data/frozen/reference/synthesis_report/results/molecules_data.json"
OUT_DIR = REPO / "figures/out"
OUT_PDF = OUT_DIR / "chemistry_case_studies.pdf"
GAPS = OUT_DIR / "GAPS.md"

LAYOUT_SEED = 12345

#: Deterministic case selection: (dataset codename in the record, graph_index).
#: Chosen after inspecting all 12 cases in the frozen record:
#: * Benzene #1693 — GEA 1.00; the most ring-dominant Benzene case: its two
#:   benzene-ring cells carry attributions (-0.35, -0.77) larger in magnitude
#:   than every bond (max |bond| = 0.36), i.e. credit concentrated at rank 2.
#:   (#9105 rejected: a 2.69 atom outlier would wash out the shared scale;
#:   #154 rejected: weaker ring cells, -0.21/-0.26.)
#: * FluorideCarbonyl #6283 — the only Fluoride-Carbonyl case whose molecule
#:   contains fluorine, so both motifs are on screen: the carbonyl C-O bond is
#:   the single largest cell (+1.38) while the fluoride F sits near zero
#:   (-0.08) — the clean separation the caption claims. (#7937/#4014 have no
#:   F atoms at all.)
#: * Mutagenicity #1675 — GEA 1.00; smallest Mutagenicity molecule (17 atoms,
#:   most readable); its 3-atom ground-truth motif holds the strongest atom
#:   (+0.96) and both strongest bonds (+0.86, +0.84) against a negative
#:   distractor branch (-0.70).
#: * AlkaneCarbonyl #466 — smallest Alkane-Carbonyl case (17 atoms) with a
#:   single compact carbonyl-plus-fluorine ground-truth motif; every
#:   top-magnitude cell lies inside or adjacent to the motif.
SELECTED = [
    ("Benzene", 1693),
    ("FluorideCarbonyl", 6283),
    ("Mutagenicity", 1675),
    ("AlkaneCarbonyl", 466),
]

#: Plain-text dataset names for panel titles (no codenames in figures).
DISPLAY_NAME = {
    "Benzene": "Benzene",
    "FluorideCarbonyl": "Fluoride-Carbonyl",
    "Mutagenicity": "Mutagenicity",
    "AlkaneCarbonyl": "Alkane-Carbonyl",
}

RC = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
}


def load_cases() -> list[dict]:
    record = json.loads(DATA.read_text())
    by_key = {(c["dataset"], c["graph_index"]): c for c in record["cases"]}
    missing = [k for k in SELECTED if k not in by_key]
    if missing:
        raise SystemExit(f"selected cases missing from frozen record: {missing}")
    return [by_key[k] for k in SELECTED]


def molecule_layout(case: dict) -> dict[int, tuple[float, float]]:
    """Deterministic 2D layout: spring_layout with a fixed seed.

    Initialized from the (deterministic) Kamada-Kawai embedding with a light
    2-iteration relaxation — pure random-init spring collapses fused hexagons,
    while this keeps rings regular and planar. Fixed seed, no wall clock.
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(case["atoms"])))
    g.add_edges_from((b["a"], b["b"]) for b in case["bonds"])
    init = nx.kamada_kawai_layout(g)
    return nx.spring_layout(g, pos=init, seed=LAYOUT_SEED, iterations=2)


def ring_polygon(ring_atoms: list[int], pos, expand: float = 1.32):
    """Ring-cell polygon vertices: atoms ordered by angle around the ring
    centroid (the record's atom list is not guaranteed cyclic), then pushed
    outward so the translucent cell reads around the atom circles."""
    pts = [pos[a] for a in ring_atoms]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [(cx + expand * (x - cx), cy + expand * (y - cy)) for x, y in pts]


def text_ink(rgba) -> str:
    """Dark ink on light atom fills, white on saturated ones."""
    r, g, b = rgba[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1a1a" if lum > 0.55 else "white"


def draw_case(ax, case: dict, cmap, norm) -> None:
    pos = molecule_layout(case)
    gt = set(case["gt_nodes"])

    # Rank-2 ring cells first (beneath everything).
    for ring in case["rings"]:
        color = cmap(norm(ring["attr"]))
        ax.add_patch(
            Polygon(
                ring_polygon(ring["atoms"], pos),
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.32,
                linewidth=1.4,
                zorder=1,
            )
        )

    # Rank-1 bonds: color and width by attribution. A thin gray underlay
    # keeps the molecular skeleton visible where attributions are near zero
    # (the diverging midpoint is near-white on a white page).
    amax = norm.vmax
    for bond in case["bonds"]:
        (x0, y0), (x1, y1) = pos[bond["a"]], pos[bond["b"]]
        lw = 1.0 + 2.6 * abs(bond["attr"]) / amax
        ax.plot(
            [x0, x1], [y0, y1],
            color="#c4c4c4", linewidth=lw + 0.9,
            solid_capstyle="round", zorder=2,
        )
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=cmap(norm(bond["attr"])),
            linewidth=lw,
            solid_capstyle="round",
            zorder=2.1,
        )

    # Rank-0 atoms: circles colored by attribution; ground-truth motif atoms
    # get a bold dark outline.
    for j, atom in enumerate(case["atoms"]):
        face = cmap(norm(atom["attr"]))
        in_gt = j in gt
        ax.scatter(
            *pos[j],
            s=215,
            facecolor=face,
            edgecolor="#111111" if in_gt else "#999999",
            linewidth=2.0 if in_gt else 0.6,
            zorder=3,
        )
        if atom["el"]:  # Mutagenicity record carries no element labels
            ax.text(
                pos[j][0],
                pos[j][1],
                atom["el"],
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_ink(face),
                zorder=4,
            )

    ax.set_aspect("equal")
    ax.set_axis_off()
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad_x = 0.14 * max(max(xs) - min(xs), 1e-9)
    pad_y = 0.14 * max(max(ys) - min(ys), 1e-9)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_title(
        f"{DISPLAY_NAME[case['dataset']]}\nGEA = {case['gea_node']:.2f}",
        fontsize=10,
    )


def note_gaps(cases: list[dict]) -> None:
    """Record data limitations without fabricating anything (append-once)."""
    notes = []
    if any(not c["element_labels"] for c in cases):
        which = ", ".join(
            DISPLAY_NAME[c["dataset"]] for c in cases if not c["element_labels"]
        )
        notes.append(
            "- `chemistry_case_studies.pdf`: the frozen record "
            "(`molecules_data.json`) carries no element labels for "
            f"{which} (`element_labels: false`, empty `el` fields), so atoms "
            "in that panel are drawn without element letters."
        )
    if not notes:
        return
    existing = GAPS.read_text() if GAPS.exists() else ""
    fresh = [n for n in notes if n not in existing]
    if fresh:
        with GAPS.open("a") as fh:
            if not existing:
                fh.write("# Known gaps in figure inputs\n\n")
            fh.write("\n".join(fresh) + "\n")


def main() -> None:
    matplotlib.rcParams.update(RC)
    cases = load_cases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared symmetric scale over every cell of every selected case.
    vmax = max(
        abs(v["attr"])
        for c in cases
        for v in (*c["atoms"], *c["bonds"], *c["rings"])
    )
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")  # diverging, neutral near-white midpoint

    fig, axes = plt.subplots(
        1, 4, figsize=(10.6, 3.35), layout="constrained"
    )
    for ax, case in zip(axes, cases):
        draw_case(ax, case, cmap, norm)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.82, pad=0.015, aspect=22)
    cbar.set_label("Shapley attribution")
    cbar.outline.set_linewidth(0.6)

    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", markersize=8.5,
                markerfacecolor="white", markeredgecolor="#111111",
                markeredgewidth=2.0, label="ground-truth motif atom",
            ),
            Patch(
                facecolor="#b0b0b0", alpha=0.4, edgecolor="#909090",
                label="ring cell (rank-2 attribution)",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    fig.savefig(OUT_PDF)
    plt.close(fig)
    note_gaps(cases)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
