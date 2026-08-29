"""Figure 1 (overview): TopoSHAP's one game, two instantiations.

Panel A (left): cells of the input complex explain a prediction.
Panel B (right): neighborhoods of the architecture explain performance.
Drawn in TopoTune's visual language: light-blue rank-0 cells, pink
rank-1 cells, burgundy rank-2 cells, soft-yellow blob highlights,
white boxes with CM math, minimal text.

Writes figures/out/overview.pdf and overview.png.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
    "svg.fonttype": "none",
})

# TopoTune palette (sampled from the published figures)
NODE_FILL = "#b9d5e9"
NODE_EDGE = "#1a1a1a"
EDGE_PINK = "#c887a3"
FACE_BURG = "#8d0045"
BLOB_FILL = "#fdf0c6"
BLOB_EDGE = "#e9c96d"
BAND_GRAY = "#e7e7e2"
DISC_GREEN = "#3ba55c"
DISC_BLUE = "#2e6fd4"
DISC_RED = "#c02f3f"
INK = "#1a1a1a"
FADE = 0.18


def blob(ax, pts, pad=0.16, **kw):
    """Soft irregular highlight blob around points (TopoTune style)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(0)
    # order by angle, push outward, smooth with a closed bezier
    ang = np.arctan2(*(pts - c).T[::-1])
    order = np.argsort(ang)
    ring = pts[order]
    out = c + (ring - c) * (1 + pad / np.maximum(
        np.linalg.norm(ring - c, axis=1), 1e-9))[:, None] + \
        (ring - c) / np.maximum(np.linalg.norm(ring - c, axis=1),
                                1e-9)[:, None] * pad
    verts, codes = [], []
    n = len(out)
    for i in range(n):
        p0, p1 = out[i], out[(i + 1) % n]
        mid = (p0 + p1) / 2
        if i == 0:
            verts.append(mid); codes.append(Path.MOVETO)
        verts.extend([p1, (p1 + out[(i + 2) % n]) / 2])
        codes.extend([Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(Path(verts, codes),
                           facecolor=kw.get("fc", BLOB_FILL),
                           edgecolor=kw.get("ec", BLOB_EDGE),
                           lw=1.4, zorder=kw.get("zorder", 0)))


def complex_glyph(ax, x, y, s=1.0, absent=(), face_alpha=1.0,
                  highlight=None):
    """The running complex: 5 rank-0, 6 rank-1, 1 rank-2 cell.

    absent: iterable of cell names to draw detached (faded, dashed).
    highlight: set of cell names drawn emphasized.
    """
    P = {  # rank-0 cells
        "a": (x - 0.85 * s, y + 0.25 * s), "b": (x, y + 0.72 * s),
        "c": (x + 0.85 * s, y + 0.25 * s), "d": (x + 0.5 * s, y - 0.62 * s),
        "e": (x - 0.5 * s, y - 0.62 * s),
    }
    EDGES = [("a", "b"), ("b", "c"), ("a", "e"), ("e", "d"),
             ("d", "c"), ("a", "c")]
    FACE = ("a", "b", "c")

    def a_of(name):
        if highlight is not None and name not in highlight:
            return FADE
        return FADE if name in absent else 1.0

    # rank-2 cell
    if "F" in absent or (highlight is not None and "F" not in highlight):
        ax.add_patch(Polygon([P[v] for v in FACE], closed=True,
                             facecolor="none", edgecolor=FACE_BURG,
                             lw=1.0, linestyle=(0, (3, 2)), alpha=0.45,
                             zorder=2))
    else:
        ax.add_patch(Polygon([P[v] for v in FACE], closed=True,
                             facecolor=FACE_BURG, edgecolor="none",
                             alpha=0.9 * face_alpha, zorder=2))
    # rank-1 cells
    for u, v in EDGES:
        name = u + v
        al = a_of(name)
        dead = name in absent or (highlight is not None
                                  and name not in highlight)
        ax.plot(*zip(P[u], P[v]), color=EDGE_PINK,
                lw=(1.2 if dead else 3.4) * s,
                alpha=0.4 if dead else 1.0, zorder=3,
                linestyle=(0, (2.5, 2)) if dead else "-",
                solid_capstyle="round")
    # rank-0 cells
    for name, (px, py) in P.items():
        dead = name in absent or (highlight is not None
                                  and name not in highlight)
        ax.add_patch(Circle((px, py), 0.15 * s,
                            facecolor="none" if dead else NODE_FILL,
                            edgecolor=NODE_EDGE, lw=0.9 if dead else 1.1,
                            linestyle=(0, (2, 2)) if dead else "-",
                            alpha=0.45 if dead else 1.0, zorder=4))
    return P


def wbox(ax, x, y, w, h, text, fs=13):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor="white", edgecolor=INK, lw=1.2,
                                zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=6)


def arrow(ax, p0, p1, curve=0.0, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={curve}",
                                 zorder=6))


def phi_bars(ax, x, y, values, colors, w=0.11, smax=0.55):
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.add_patch(plt.Rectangle((x + i * (w + 0.045), y), w,
                                   smax * v, facecolor=c,
                                   edgecolor=INK, lw=0.8, zorder=5))
    ax.plot([x - 0.06, x + len(values) * (w + 0.045)], [y, y],
            color=INK, lw=1.0, zorder=5)


def hasse_icon(ax, x, y, s=0.42, color=EDGE_PINK, active=True):
    """Small per-neighborhood Hasse-graph icon."""
    al = 1.0 if active else FADE
    pts = [(x - s, y), (x, y + 0.7 * s), (x + s, y), (x, y - 0.7 * s)]
    for i in range(4):
        for j in range(i + 1, 4):
            if (i + j) % 2:
                ax.annotate("", pts[j], pts[i], zorder=4,
                            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                            color="#777777", alpha=al,
                                            mutation_scale=7))
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.10, facecolor=color,
                            edgecolor=INK, lw=0.9, alpha=al, zorder=5))




def rank_glyph(ax, x, y, rank, s=1.0, alpha=1.0, zorder=5):
    """Tiny pictogram of a cell of the given rank."""
    if rank == 0:
        ax.add_patch(Circle((x, y), 0.085 * s, facecolor=NODE_FILL,
                            edgecolor=INK, lw=0.9, alpha=alpha,
                            zorder=zorder))
    elif rank == 1:
        ax.plot([x - 0.11 * s, x + 0.11 * s], [y, y], color=EDGE_PINK,
                lw=3.2 * s, alpha=alpha, zorder=zorder,
                solid_capstyle="round")
    else:
        ax.add_patch(Polygon([(x - 0.11 * s, y - 0.08 * s),
                              (x + 0.11 * s, y - 0.08 * s),
                              (x, y + 0.1 * s)], closed=True,
                             facecolor=FACE_BURG, edgecolor="none",
                             alpha=0.92 * alpha, zorder=zorder))


def nb_icon(ax, x, y, src, dst, s=1.0, alpha=1.0):
    """Neighborhood pictogram: source rank -> destination rank."""
    rank_glyph(ax, x - 0.22 * s, y, src, s=s, alpha=alpha)
    ax.add_patch(FancyArrowPatch((x - 0.1 * s, y), (x + 0.1 * s, y),
                                 arrowstyle="-|>", mutation_scale=8 * s,
                                 lw=1.0, color=INK, alpha=alpha,
                                 zorder=5))
    rank_glyph(ax, x + 0.22 * s, y, dst, s=s, alpha=alpha)


def bracket(ax, x, y_top, y_bot, out_len=0.22):
    """Right square bracket collecting a column, with an output stem."""
    tick = 0.1
    ax.plot([x - tick, x, x, x - tick], [y_top, y_top, y_bot, y_bot],
            color=INK, lw=1.3, zorder=5)
    mid = (y_top + y_bot) / 2
    ax.plot([x, x + out_len], [mid, mid], color=INK, lw=1.3, zorder=5)
    return (x + out_len, mid)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.6, 4.5))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # ================= PANEL A: cells explain a prediction ==========
    ax.text(0.25, 4.25, "A.", fontsize=15, fontweight="bold")
    ax.text(1.05, 4.25, "explaining a prediction", fontsize=12.5,
            style="italic")

    complex_glyph(ax, 1.3, 2.75, s=0.75)
    ax.text(1.3, 1.7, r"$\mathcal{C}$", fontsize=15, ha="center")
    ax.text(1.3, 1.38, "players = cells", fontsize=10.5, ha="center")

    coal = [(("b", "ab", "bc", "F"), 3.5, DISC_GREEN, 0.13),
            (("d", "ed", "dc"), 2.6, "#7fc97f", 0.10),
            (("e", "ae", "ed", "ac", "F"), 1.7, "#b9e2b9", 0.08)]
    for absent, y, dc, dr in coal:
        Pc = complex_glyph(ax, 3.35, y, s=0.42, absent=absent)
        present = [Pc[k] for k in "abcde" if k not in absent]
        blob(ax, present, pad=0.22)
        arrow(ax, (4.05, y), (4.42, y))
        ax.add_patch(Circle((5.62, y), dr, facecolor=dc,
                            edgecolor=INK, lw=1.0, zorder=6))
    ax.text(3.35, 1.06, r"$\vdots$", fontsize=13, ha="center")
    ax.text(3.35, 0.7, r"coalitions $\mathcal{C}[S]$", fontsize=10.5,
            ha="center")
    arrow(ax, (2.18, 2.75), (2.6, 2.75))

    wbox(ax, 4.95, 2.6, 0.92, 2.45, "", fs=15)
    ax.text(4.95, 2.6, r"$f_\theta$", fontsize=16, ha="center",
            va="center", zorder=7)
    for _, y, _, _ in coal:
        arrow(ax, (5.42, y), (5.5, y))
    ax.text(5.62, 3.92, r"$v(S)$", fontsize=12.5, ha="center")

    # bracket collects the game values; two labeled outputs
    tipA = bracket(ax, 6.0, 3.62, 1.58)
    arrow(ax, tipA, (6.95, 3.35), curve=-0.15)
    ax.text(6.55, 3.6, "average marginals", fontsize=9, ha="center",
            style="italic", rotation=24)
    arrow(ax, tipA, (6.95, 1.85), curve=0.15)
    ax.text(6.55, 1.72, "greedy prune", fontsize=9, ha="center",
            style="italic", rotation=-24)

    # output 1: attribution per cell of C (bars grouped by rank,
    # rank pictograms under the axis)
    ax.text(7.25, 4.25, "output 1", fontsize=10.5, style="italic",
            color="#666666")
    phi_vals = [0.55, 0.72, 0.6, 0.18, 0.15,
                0.62, 0.58, 0.14, 0.10, 0.13, 0.66,
                1.0]
    phi_cols = [NODE_FILL] * 5 + [EDGE_PINK] * 6 + [FACE_BURG]
    bx, bw = 7.15, 0.085
    phi_bars(ax, bx, 3.3, phi_vals, phi_cols, w=bw, smax=0.55)
    step = bw + 0.045
    for center, rk in ((bx + 2 * step, 0), (bx + 7.5 * step, 1),
                       (bx + 11 * step, 2)):
        rank_glyph(ax, center + bw / 2, 3.12, rk, s=0.9)
    ax.text(9.0, 3.55, r"$\phi_\sigma$", fontsize=13.5, ha="center")
    ax.text(7.95, 2.72, r"attribution per cell of $\mathcal{C}$",
            fontsize=10.5, ha="center")

    # output 2: the sufficient sub-complex of C
    ax.text(7.25, 0.62, "output 2", fontsize=10.5, style="italic",
            color="#666666")
    complex_glyph(ax, 7.75, 1.5, s=0.5, highlight={"a", "b", "c",
                                                   "ab", "bc", "ac",
                                                   "F"})
    ax.text(8.8, 1.75, r"$S^{*}_{b} \subset \mathcal{C}$",
            fontsize=13, ha="center")
    ax.text(8.35, 0.9, "sufficient sub-complex", fontsize=10.5,
            ha="center")

    # ================= divider =====================================
    ax.plot([9.55, 9.55], [0.5, 4.25], color="#bbbbbb", lw=1.0,
            linestyle=(0, (4, 3)))

    # ============ PANEL B: neighborhoods explain performance =======
    ax.text(9.8, 4.25, "B.", fontsize=15, fontweight="bold")
    ax.text(10.55, 4.25, "explaining performance", fontsize=12.5,
            style="italic")

    # players: neighborhoods as source-rank -> destination-rank icons
    NBS = [(0, 0), (1, 2), (2, 1)]
    nys = [3.35, 2.6, 1.85]
    for y, (src, dst), lab in zip(nys, NBS, [r"$\mathcal{N}_1$",
                                             r"$\mathcal{N}_2$",
                                             r"$\mathcal{N}_3$"]):
        ax.text(9.95, y - 0.035, lab, fontsize=11.5, ha="center")
        nb_icon(ax, 10.55, y, src, dst, s=1.0)
    ax.text(10.3, 1.14, "players =", fontsize=10.5, ha="center")
    ax.text(10.3, 0.86, "neighborhoods", fontsize=10.5, ha="center")

    # coalitions of neighborhoods (excluded ones faded)
    def ncoalition(x, y, active):
        for i, act in enumerate(active):
            py = y + (1 - i) * 0.28
            nb_icon(ax, x, py, *NBS[i], s=0.55,
                    alpha=1.0 if act else 0.15)
        blob(ax, [(x - 0.3, y - 0.34), (x - 0.3, y + 0.34),
                  (x + 0.3, y - 0.34), (x + 0.3, y + 0.34)], pad=0.1)
    coalsB = [((True, False, True), 3.5, DISC_BLUE, 0.125),
              ((True, True, False), 2.6, "#6f9fe0", 0.10),
              ((False, True, True), 1.7, "#b3c9ec", 0.08)]
    for act, y, dc, dr in coalsB:
        ncoalition(11.55, y, act)
        arrow(ax, (12.0, y), (12.2, y))
        ax.add_patch(Circle((13.25, y), dr, facecolor=dc,
                            edgecolor=INK, lw=1.0, zorder=6))
    ax.text(11.55, 1.06, r"$\vdots$", fontsize=13, ha="center")
    ax.text(11.55, 0.7, "coalitions", fontsize=10.5, ha="center")

    wbox(ax, 12.62, 2.6, 0.7, 2.45, "", fs=12)
    ax.text(12.62, 2.74, "stem", fontsize=10, ha="center", zorder=7)
    ax.text(12.62, 2.47, r"$f_\theta$", fontsize=12.5, ha="center",
            zorder=7)
    for _, y, _, _ in coalsB:
        arrow(ax, (13.0, y), (13.08, y))
    ax.text(13.25, 3.92, r"$v(S) = \mathrm{acc}$", fontsize=11.5,
            ha="center")

    # bracket + single output
    tipB = bracket(ax, 13.58, 3.62, 1.58)
    arrow(ax, tipB, (13.95, 2.05), curve=0.2)
    ax.text(14.22, 2.66, "average marginals", fontsize=9,
            ha="center", style="italic", rotation=-65)
    phi_bars(ax, 13.72, 1.42, [0.95, 0.2, 0.7],
             [NODE_FILL, FACE_BURG, EDGE_PINK], w=0.13, smax=0.5)
    for i, (src, dst) in enumerate(NBS):
        nb_icon(ax, 13.72 + 0.065 + i * 0.175, 1.24, src, dst, s=0.32)
    ax.text(14.5, 1.62, r"$\phi_{\mathcal{N}}$", fontsize=13,
            ha="center")
    ax.text(13.35, 0.86, "attribution of performance",
            fontsize=10.5, ha="left")
    ax.text(13.35, 0.6, "per neighborhood", fontsize=10.5, ha="left")

    fig.tight_layout(pad=0.3)
    import pathlib
    out = pathlib.Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "overview.pdf", bbox_inches="tight")
    fig.savefig(out / "overview.png", dpi=180, bbox_inches="tight")
    print("wrote", out / "overview.pdf")


if __name__ == "__main__":
    main()
