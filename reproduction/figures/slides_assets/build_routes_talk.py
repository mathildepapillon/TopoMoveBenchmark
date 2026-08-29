"""Talk figure: which cells participate, per explained model.

One column per model in the participation attribution figure
(build_participation_pair.py), same order, so the two slides mirror
each other 1:1. Solid arrows = backbone message routes (neighborhoods);
dashed arrows = the readout's path to the logits; filled boxes = the
rank participates in the prediction (its cells can earn credit).

Output: figures/slides_assets/routes_talk.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "routes_talk.png"

NAVY = "#33487c"
RANK_Y = {2: 0.80, 1: 0.55, 0: 0.30}
LOGIT_Y = 0.075
RANK_LABEL = {2: "triangles (rank 2)", 1: "edges (rank 1)",
              0: "vertices (rank 0)"}
BOX_W, BOX_H = 0.13, 0.085


def box(ax, cx, rank):
    y = RANK_Y[rank]
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.008", fc=NAVY, ec=NAVY, lw=1.6))
    ax.text(cx, y, RANK_LABEL[rank], ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white")


def logits_node(ax, cx, label="logits"):
    ax.add_patch(FancyBboxPatch(
        (cx - 0.042, LOGIT_Y - 0.036), 0.084, 0.072,
        boxstyle="round,pad=0.008", fc="0.94", ec="0.6", lw=1.2))
    ax.text(cx, LOGIT_Y, label, ha="center", va="center",
            fontsize=10.5, style="italic", color="0.25")


def arrow(ax, p0, p1, rad=0.0, dashed=False, lw=2.2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=15, lw=lw, color=NAVY,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=2, shrinkB=2))


def self_loop(ax, cx, rank, side="left"):
    y = RANK_Y[rank]
    x0 = cx - BOX_W / 2 - 0.006 if side == "left" else cx + BOX_W / 2 + 0.006
    rad = 2.4 if side == "left" else -2.4
    arrow(ax, (x0, y + 0.022), (x0, y - 0.022), rad=rad, lw=2.0)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.0, 4.4), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    centers = [0.125, 0.375, 0.625, 0.875]
    titles = [
        "CWN — native readout",
        "GCCN — ladder pick (seed 43)",
        "GCCN — anchored pick",
        "HOPSE — sweep pick",
    ]
    for cx, t in zip(centers, titles):
        ax.text(cx, 0.97, t, ha="center", va="center", fontsize=13.5)
        for r in (2, 1, 0):
            box(ax, cx, r)
        logits_node(ax, cx)

    # CWN native: routes 0->1, 1->1 (loop), 2->1; readout cascades
    # 2->1->0 on the right, then rank 0 -> logits
    cx = centers[0]
    xl = cx - BOX_W / 2 - 0.012
    arrow(ax, (xl, RANK_Y[0] + 0.03), (xl, RANK_Y[1] - 0.03), rad=0.85)
    arrow(ax, (xl, RANK_Y[2] - 0.03), (xl, RANK_Y[1] + 0.03), rad=-0.85)
    self_loop(ax, cx, 1, side="right")
    xr = cx + BOX_W / 2 + 0.03
    arrow(ax, (xr, RANK_Y[2]), (xr, RANK_Y[1] + 0.02), rad=-0.55, dashed=True,
          lw=1.8)
    arrow(ax, (xr, RANK_Y[1] - 0.02), (xr, RANK_Y[0]), rad=-0.55, dashed=True,
          lw=1.8)
    arrow(ax, (cx, RANK_Y[0] - BOX_H / 2 - 0.01), (cx, LOGIT_Y + 0.045),
          dashed=True, lw=1.8)
    ax.text(xr + 0.026, RANK_Y[1], "readout\npushes\nsignal\ndown",
            ha="left", va="center", fontsize=8.5, style="italic",
            color="0.35")

    # ladder: single composed route 2->0, head reads rank 0
    cx = centers[1]
    xr = cx + BOX_W / 2 + 0.004
    arrow(ax, (xr, RANK_Y[2] - 0.02), (xr, RANK_Y[0] + 0.02), rad=-0.5)
    arrow(ax, (cx, RANK_Y[0] - BOX_H / 2 - 0.01), (cx, LOGIT_Y + 0.045),
          dashed=True, lw=1.8)

    # anchored: 0->0 (loop), 1->0, 2->0, head reads rank 0
    cx = centers[2]
    xr = cx + BOX_W / 2 + 0.004
    arrow(ax, (xr + 0.006, RANK_Y[2] - 0.02), (xr + 0.006, RANK_Y[0] + 0.02),
          rad=-0.5)
    arrow(ax, (xr, RANK_Y[1] - 0.02), (xr, RANK_Y[0] + 0.02), rad=-0.4)
    self_loop(ax, cx, 0, side="left")
    arrow(ax, (cx, RANK_Y[0] - BOX_H / 2 - 0.01), (cx, LOGIT_Y + 0.045),
          dashed=True, lw=1.8)

    # HOPSE: every rank pooled into the head
    cx = centers[3]
    arrow(ax, (cx + BOX_W / 2 + 0.004, RANK_Y[2] - 0.02),
          (cx + 0.035, LOGIT_Y + 0.045), rad=-0.35, dashed=True, lw=1.8)
    arrow(ax, (cx - BOX_W / 2 - 0.004, RANK_Y[1] - 0.02),
          (cx - 0.035, LOGIT_Y + 0.045), rad=0.3, dashed=True, lw=1.8)
    arrow(ax, (cx, RANK_Y[0] - BOX_H / 2 - 0.01), (cx, LOGIT_Y + 0.045),
          dashed=True, lw=1.8)
    logits_node(ax, cx, label="logits")
    ax.text(cx, 0.905, "per-cell encodings are functions of the whole complex",
            ha="center", fontsize=9.5, color="0.35", style="italic")

    fig.text(0.5, 0.015,
             "filled = participates in the prediction (earns credit)   ·   "
             "solid arrows = message routes   ·   dashed = readout path",
             ha="center", va="bottom", fontsize=11, color="0.45")

    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
