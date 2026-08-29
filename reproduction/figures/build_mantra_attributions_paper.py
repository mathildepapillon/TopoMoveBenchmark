"""MANTRA attribution figure, paper format (single-column \\linewidth).

Paper-format sibling of figures/build_mantra_attr_figure.py (the deck
version). Data loading and label-flip semantics are copied from that
builder verbatim; content is identical, only sizing/typography changes.

Inputs (both frozen, read-only):
  data/frozen/mantra_attr_fig.json      -- job 419349 (menu/GCCN valid;
                                           its HOPSE game was degenerate --
                                           those hopse entries are all-zero
                                           and are NOT used here)
  data/frozen/mantra_attr_hopse.json    -- job 419487 rerun at the encoding
                                           interface (HopseCellMaskingGame);
                                           the ONLY source of HOPSE phi

LABEL FLIP (single authoritative site, same as the deck builder): job
419349 ran with pos_cls = majority before the label-direction fix, so in
BOTH files the role named "orientable" is truly the NON-orientable complex
(a Klein bottle; test index 73) and "non_orientable" is truly the
ORIENTABLE one (a torus; test index 243). The recorded margin is
logit[non-orientable] - logit[orientable]; this builder NEGATES phi so
every bar reads "contribution to the orientability score".

Y-SCALE CHOICE (both variants were rendered and inspected):
  (a) uniform y-span across all three panels -- HOPSE's largest |phi| is
      ~4.7 vs ~0.23 for menu/GCCN, so under one span the menu/GCCN bars
      collapse to <4% of the half-span (sub-millimetre at print size) and
      the panels' evidence (uniform vertex credit; triangles-only credit)
      disappears. Rejected as unreadable.
  (b) menu and GCCN share one y-limit (their scales are comparable, so
      those two panels stay directly comparable); HOPSE sits on its own
      scale with the ratio stated in its axis label. KEPT.
Linear axes throughout (asserted). Per-cell bars are single Shapley values
from one estimation run -- no aggregation happens, so no error bars.

Output: figures/out/mantra_attributions.pdf (vector, ~10.8 in wide for
\\linewidth use, serif). Seed-43 case study (disclosed in the caption);
estimator: 512 permutation passes, zeros baseline, ~26k unique
evaluations per game.
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
OUT = REPO / "figures" / "out" / "mantra_attributions.pdf"

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

# Paper typography: fonts sized for a ~10.8 in figure scaled to a ~5.5 in
# \linewidth (factor ~0.51), landing at ~6-8 pt effective. Serif to blend
# with the LaTeX body; Type 42 fonts for camera-ready portability.
RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
}


def load_pair():
    """Identical to the deck builder's load_pair (label flip included)."""
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


def build(uniform_y: bool = False, out: Path = OUT) -> Path:
    matplotlib.rcParams.update(RC)
    main_json, pair = load_pair()

    stats = {}
    for model, _ in MODELS:
        m = main_json["models"][model]
        stats[model] = (m["test_auroc"], m["count_conditional_auroc"])

    # The matched pair: identical cell counts, so one x-layout serves all.
    counts = list(main_json["pair"]["counts"])
    assert counts == [9, 27, 18], f"unexpected pair counts {counts}"
    ranks = pair["torus"]["menu"]["ranks"]
    for t in pair:
        for m, _ in MODELS:
            assert np.array_equal(pair[t][m]["ranks"], ranks), \
                "player ranks differ across games -- x-layout invalid"
    n = len(ranks)
    bounds = [0] + [int(np.searchsorted(ranks, r + 1)) for r in (0, 1, 2)]
    assert [b1 - b0 for b0, b1 in zip(bounds[:-1], bounds[1:])] == counts
    band_labels = ["vertices", "edges", "triangles"]

    # Per-panel half-spans (symmetric about zero, 1.35x headroom for the
    # band labels); menu and GCCN share theirs unless uniform_y forces one
    # span everywhere. See the module docstring for the readability call.
    HEADROOM = 1.35
    raw = {m: max(np.abs(pair[t][m]["phi"]).max() for t in pair)
           for m, _ in MODELS}
    if uniform_y:
        ymaxes = {m: HEADROOM * max(raw.values()) for m in raw}
    else:
        shared = HEADROOM * max(raw["menu"], raw["gccn"])
        ymaxes = {"menu": shared, "gccn": shared,
                  "hopse": HEADROOM * raw["hopse"]}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=False,
                             constrained_layout=True)
    x = np.arange(n)
    for ax, (model, title) in zip(axes, MODELS):
        ymax = ymaxes[model]
        for b0, b1, lab in zip(bounds[:-1], bounds[1:], band_labels):
            if (b1 - b0) and (bounds.index(b0) % 2 == 1):
                ax.axvspan(b0 - 0.5, b1 - 0.5, color="0.92", zorder=0)
            ax.text((b0 + b1 - 1) / 2, -ymax * 0.93, lab, ha="center",
                    fontsize=11.5, color="0.35")
        for truth, off in (("torus", -0.21), ("klein", 0.21)):
            g = pair[truth][model]
            # Single Shapley values per cell -- nothing aggregated, so no
            # error bars (repo rule: bars only where aggregation happens).
            ax.bar(x + off, g["phi"], width=0.4, color=COLORS[truth],
                   label=LABELS[truth], zorder=2)
        ax.axhline(0.0, lw=0.8, color="0.4", zorder=1)
        auroc, cc = stats[model]
        # Three short title lines: at \linewidth each panel prints ~1.8 in
        # wide, so the deck's single long stat line would need a <5 pt
        # effective font -- wrapping keeps every line legible.
        ax.set_title(f"{title}\ntest AUROC {auroc:.2f}\n"
                     f"matched-count AUROC {cc:.3f}")
        sums = {t: pair[t][model]["phi"].sum() for t in ("torus", "klein")}
        ax.text(0.99, 0.97,
                f"Σφ torus {sums['torus']:+.2f}\n"
                f"Σφ Klein {sums['klein']:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=12,
                linespacing=1.4)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("cell (rank-major order)")
        # Linear axes are a hard repo requirement -- never log-scale here.
        assert ax.get_xscale() == "linear" and ax.get_yscale() == "linear"
    axes[0].set_ylabel("contribution to\norientability score")
    if not uniform_y:
        # HOPSE's own scale, with the difference stated on the axis.
        ratio = ymaxes["hopse"] / ymaxes["menu"]
        axes[2].set_ylabel(f"own y-scale\n({ratio:.0f}x left panels)")
    # One figure-level legend above the panels: inside panel 1 it would
    # collide with the sum-of-phi annotation at paper panel widths.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="outside upper center",
               ncol=2, handlelength=1.4, columnspacing=2.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    out = build()
    print(f"wrote {out}")

    # Verification prints (same numbers the deck builder reports).
    main_json, pair = load_pair()
    print("\npanel-title numbers (test AUROC / matched-count AUROC):")
    for model, title in MODELS:
        m = main_json["models"][model]
        print(f"  {title}: {m['test_auroc']:.2f} / "
              f"{m['count_conditional_auroc']:.3f}")
    for truth in ("torus", "klein"):
        row = " ".join(
            f"{m}:sum_phi={pair[truth][m]['phi'].sum():+.3f}"
            f"(v_full={pair[truth][m]['v_full']:+.3f})"
            for m, _ in MODELS)
        print(f"  {truth}: {row}")


if __name__ == "__main__":
    main()
