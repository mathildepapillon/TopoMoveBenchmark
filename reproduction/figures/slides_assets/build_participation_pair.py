"""Talk figure: participation-game attributions, four models, one pair.

Inputs (results/mantra_attr_fig/, post-freeze presentation outputs, all
computed under the participation game with pool-faithful readouts):
  cwn_native_pair.json          -- CWN neighborhoods trained with the
                                   NATIVE PropagateSignalDown readout
  participation_pair.json       -- ladder pick + anchored pick (NoReadOut)
  hopse_participation_pair.json -- HOPSE sweep pick (encodings recomputed
                                   from the detached complex per coalition)

All four record margin = logit[non-orientable] - logit[orientable] on the
matched-count test pair (torus idx 243, Klein idx 73; 9/27/18 cells).
This builder NEGATES phi so every bar reads "contribution to the
orientability score".

Output: figures/slides_assets/participation_pair.png + stdout table.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results" / "mantra_attr_fig"
OUT = Path(__file__).resolve().parent / "participation_pair.png"

COLORS = {"torus": "#1f3d99", "klein": "#a34808"}
LABELS = {"torus": "torus (orientable)",
          "klein": "Klein bottle (non-orientable)"}
BAND_LABELS = ["vertices", "edges", "triangles"]


def load_panels():
    native = json.loads((R / "cwn_native_pair.json").read_text())
    pair = json.loads((R / "participation_pair.json").read_text())
    hopse = json.loads((R / "hopse_participation_pair.json").read_text())
    return [
        (native, "CWN — native readout",
         f"signal propagates down to the head · test AUROC "
         f"{native['test_auroc']:.2f}"),
        (pair["gccn"], "GCCN — ladder pick (seed 43)",
         "every rank participates: signal, conduits, pooling"),
        (pair["anchored"], "GCCN — anchored pick",
         "every rank participates"),
        (hopse, "HOPSE — sweep pick",
         "encodings recomputed from the detached complex"),
    ]


def main() -> None:
    panels = load_panels()

    ranks = np.asarray(panels[0][0]["torus"]["player_ranks"], dtype=int)
    n = len(ranks)
    bounds = [0] + [int(np.searchsorted(ranks, r + 1)) for r in (0, 1, 2)]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.6), dpi=200,
                             sharey=False, constrained_layout=True)
    for ax, (entry, title, subtitle) in zip(axes, panels):
        phis = {t: -np.asarray(entry[t]["phi"], dtype=float)
                for t in ("torus", "klein")}
        ymax = max(np.abs(p).max() for p in phis.values()) * 1.45
        for b0, b1, lab in zip(bounds[:-1], bounds[1:], BAND_LABELS):
            if (b1 - b0) and (bounds.index(b0) % 2 == 1):
                ax.axvspan(b0 - 0.5, b1 - 0.5, color="0.92", zorder=0)
            ax.text((b0 + b1 - 1) / 2, -ymax * 0.90, lab, ha="center",
                    fontsize=8.5, color="0.35")
            for truth, dy in (("torus", 0.965), ("klein", 0.83)):
                s = phis[truth][b0:b1].sum()
                ax.text((b0 + b1 - 1) / 2, ymax * dy, f"Σ {s:+.2f}",
                        ha="center", va="top", fontsize=8.5,
                        color=COLORS[truth])
        for truth, off in (("torus", -0.21), ("klein", 0.21)):
            ax.bar(x + off, phis[truth], width=0.4, color=COLORS[truth],
                   label=LABELS[truth], zorder=2)
        ax.axhline(0.0, lw=0.8, color="0.4", zorder=1)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("contribution to orientability", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=2, frameon=False,
               loc="outside upper left")
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")

    print("\nper-rank sums (contribution to orientability):")
    for entry, title, _ in panels:
        for truth in ("torus", "klein"):
            phi = -np.asarray(entry[truth]["phi"], dtype=float)
            sums = [phi[b0:b1].sum() for b0, b1 in zip(bounds, bounds[1:])]
            print(f"  {title} / {truth}: "
                  + " ".join(f"{lab}={s:+.3f}"
                             for lab, s in zip(BAND_LABELS, sums))
                  + f"  v_full={-entry[truth]['v_full']:+.3f}")


if __name__ == "__main__":
    main()
