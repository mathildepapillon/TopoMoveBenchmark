"""MANTRA attribution figure: three models, one matched-count test pair.

Inputs (both frozen):
  data/frozen/mantra_attr_fig.json      -- job 419349 (menu/GCCN valid;
                                           its HOPSE game was degenerate)
  data/frozen/mantra_attr_hopse.json    -- job 419487 rerun at the encoding
                                           interface (HopseCellMaskingGame)

LABEL FLIP (single authoritative site): job 419349 ran with
pos_cls = majority before the label-direction fix, so in BOTH files the
role named "orientable" is truly the NON-orientable complex (a Klein
bottle; test index 73) and "non_orientable" is truly the ORIENTABLE one
(a torus; test index 243). The recorded margin is
logit[non-orientable] - logit[orientable]; this builder NEGATES phi so
every bar reads "contribution to the orientability score".

Output: figures/mantra_attr_pair.png plus the slide-table numbers on
stdout. Seed-43 case study (disclosed on the slide); estimator: 512
permutation passes, zeros baseline, ~26k unique evaluations per game.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "data" / "frozen" / "mantra_attr_fig.json"
HOPSE = REPO / "data" / "frozen" / "mantra_attr_hopse.json"
OUT = REPO / "figures" / "mantra_attr_pair.png"

# True semantics of the JSON role names (see module docstring).
ROLE_TO_TRUTH = {"orientable": "klein", "non_orientable": "torus"}
COLORS = {"torus": "#1f3d99", "klein": "#a34808"}
LABELS = {"torus": "torus (orientable)",
          "klein": "Klein bottle (non-orientable)"}
MODELS = [
    ("menu", "TopoBench menu (val-best)"),
    ("gccn", "GCCN (ladder-selected)"),
    ("hopse", "HOPSE (sweep-best)"),
]


def load_pair():
    main = json.loads(MAIN.read_text())
    hopse = json.loads(HOPSE.read_text())
    pair = {}
    for role, truth in ROLE_TO_TRUTH.items():
        entry = {}
        for model, _ in MODELS:
            src = hopse[role] if model == "hopse" else main["pair"][role][model]
            entry[model] = {
                # Negate: recorded margin is toward NON-orientability.
                "phi": -np.asarray(src["phi"], dtype=float),
                "ranks": np.asarray(src["player_ranks"], dtype=int),
                "v_full": -float(src["v_full"]),
                "v_empty": -float(src["v_empty"]),
            }
        pair[truth] = entry
    return main, pair


def main() -> None:
    main_json, pair = load_pair()

    stats = {}
    for model, _ in MODELS:
        m = main_json["models"][model]
        stats[model] = (m["test_auroc"], m["count_conditional_auroc"],
                        m["surrogate_agreement"])

    ranks = pair["torus"]["menu"]["ranks"]
    n = len(ranks)
    bounds = [0] + [int(np.searchsorted(ranks, r + 1)) for r in (0, 1, 2)]
    band_labels = ["vertices", "edges", "triangles"]

    # Per-panel y-spans: HOPSE's credit is ~20x menu/GCCN's, and the
    # evidence in the small panels is the PATTERN (uniform vertex credit;
    # triangles-only credit). Scales are printed on every axis and the
    # per-complex totals are annotated in data units, so nothing hides.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4), dpi=200,
                             sharey=False, constrained_layout=True)
    x = np.arange(n)
    for ax, (model, title) in zip(axes, MODELS):
        ymax = max(np.abs(pair[t][model]["phi"]).max() for t in pair) * 1.3
        for b0, b1, lab in zip(bounds[:-1], bounds[1:], band_labels):
            if (b1 - b0) and (bounds.index(b0) % 2 == 1):
                ax.axvspan(b0 - 0.5, b1 - 0.5, color="0.92", zorder=0)
            ax.text((b0 + b1 - 1) / 2, -ymax * 0.93, lab, ha="center",
                    fontsize=8, color="0.35")
        for truth, off in (("torus", -0.21), ("klein", 0.21)):
            g = pair[truth][model]
            ax.bar(x + off, g["phi"], width=0.4, color=COLORS[truth],
                   label=LABELS[truth], zorder=2)
        ax.axhline(0.0, lw=0.8, color="0.4", zorder=1)
        auroc, cc, surr = stats[model]
        ax.set_title(f"{title}\ntest AUROC {auroc:.2f} · matched-count "
                     f"AUROC {cc:.3f}", fontsize=10)
        sums = {t: pair[t][model]["phi"].sum() for t in ("torus", "klein")}
        ax.text(0.99, 0.97,
                f"Σφ torus {sums['torus']:+.2f}\n"
                f"Σφ Klein {sums['klein']:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                linespacing=1.4)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("cell (rank-major order)", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("contribution to orientability score", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")

    print("\nslide-table numbers (test AUROC / matched-count AUROC / "
          "count-surrogate agreement):")
    for model, title in MODELS:
        a, c, s = stats[model]
        print(f"  {title}: {a:.2f} / {c:.3f} / {s:.3f}")
    for truth in ("torus", "klein"):
        row = " ".join(
            f"{m}:sum_phi={pair[truth][m]['phi'].sum():+.3f}"
            f"(v_full={pair[truth][m]['v_full']:+.3f})"
            for m, _ in MODELS)
        print(f"  {truth}: {row}")


if __name__ == "__main__":
    main()
