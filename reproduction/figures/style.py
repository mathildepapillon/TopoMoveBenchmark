"""Shared style for TopoSHAP figures — encodes the lead-figure spec.

Marker vocabulary (binding, from fig-a09699c2f95b):

* per-seed runs: open circles.
* one star per fixed k — each a TRUE one-run recipe, no best-of-k selection:
  k=2 ember five-point star, k=3 slate-blue three-point (triangle) star,
  k=6 dark four-point (diamond) star.
* menu: a single filled square at its true cost, symmetric per-seed-selection
  error bars.
* ceilings: dotted dark line = exact 6-space optimum; light gray line =
  best-known 11-space lower bound.

Hard requirements enforced by the builder, not just documented: linear axes,
uniform y-span across panels, error bars on every aggregated marker.
"""

from __future__ import annotations

EMBER = "#d95f02"
SLATE_BLUE = "#6a7fdb"
DARK = "#222222"
GRAY_LIGHT = "#bbbbbb"
SEED_DOT = "#7a7a7a"
MENU_SQUARE = "#444444"

#: k -> (matplotlib marker, color, label). Star family per spec; the k=3
#: star is OURS in the GCCN space and carries a dark, saturated blue.
STAR_BY_K = {
    2: ("*", EMBER, "recipe k=2"),
    3: ((3, 1, 0), "#1f3d99", "OURS (GCCN)"),
    6: ((4, 1, 0), DARK, "recipe k=6"),
}

STAR_SIZE = 160
DOT_SIZE = 28
SQUARE_SIZE = 70

RC = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
}


def apply() -> None:
    import matplotlib

    matplotlib.rcParams.update(RC)


def uniform_yspan(panel_ranges: list[tuple[float, float]], pad: float = 0.06):
    """Per-panel (ymin, ymax) sharing one span: the max data range + padding.

    Each panel is centered on its own data but every panel's ylim covers the
    same extent, so vertical distances are comparable across panels.
    """
    spans = [hi - lo for lo, hi in panel_ranges]
    span = max(spans) * (1 + 2 * pad)
    out = []
    for lo, hi in panel_ranges:
        mid = (lo + hi) / 2
        out.append((mid - span / 2, mid + span / 2))
    return out
