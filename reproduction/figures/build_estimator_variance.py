#!/usr/bin/env python
"""Build the estimator-variance appendix figure (fig:app-estimator-variance).

Sampled-estimator error vs. number of permutation passes, per dataset,
from the frozen #12 cheap-value simulation (data/frozen/cheap_sim12.json).

Metric (the paper's committed error measure, cf. \\ResRegretMax): the
cheap-game accuracy regret of the sampled estimator's pick against the
exact-game selection. For each dataset, training seed, subset size
k in {2, 3, 6} and pass budget in {32, 64, 128, 256, 512}, the frozen
record replays the backward-elimination selection with 50 permutation
resamples of the attribution estimator and stores the mean cheap-value
regret of the resampled picks relative to the exact-game (2048 unique
coalition evaluations) pick — `per_seed[seed].permutation[budget]
.per_k[k].mean_cheap_regret`. Positive = the sampled pick scores below
the exact-game pick on the masked-validation game; negative = the
resampled path found a higher-valued subset than the exact-path pick.

Rendered: one panel per dataset (small multiples), one line per fixed k
(the repo's k marker/color vocabulary), marker = mean across the 3
training seeds, error bar = sd across seeds, on EVERY marker. Linear
axes, uniform y-span across panels, production budget (128 passes)
marked with a vertical reference line. No fabricated points: the script
fails loudly if any (dataset, seed, budget, k) cell is missing, and it
cross-checks the per-seed means it plots against the frozen aggregated
values.

Input (read-only): data/frozen/cheap_sim12.json
Output: figures/out/estimator_variance.pdf (+ .png preview)

Usage:
    python figures/build_estimator_variance.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data" / "frozen" / "cheap_sim12.json"
OUT_DIR = REPO / "figures" / "out"

#: panel order (graph-level datasets first, then node-level), matching the
#: original record's figure ordering (reference/figure-sources)
ORDER = [
    "benzene",
    "mutagenicity_gxai",
    "NCI1",
    "fluoride_carbonyl",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
]

DISPLAY = {
    "benzene": "Benzene",
    "mutagenicity_gxai": "Mutagenicity",
    "NCI1": "NCI1",
    "fluoride_carbonyl": "Fluoride-Carbonyl",
    "cocitation_cora": "Cora",
    "cocitation_citeseer": "Citeseer",
    "cocitation_pubmed": "Pubmed",
}

SEEDS = ("42", "43", "44")
KS = ("2", "3", "6")
BUDGETS = (32, 64, 128, 256, 512)
PRODUCTION_PASSES = 128  # \ResSampledPasses — the validated attribution budget

#: k -> (marker, color, label); colors follow the repo's k vocabulary
#: (style.STAR_BY_K), marker shapes differ per k as secondary encoding
K_STYLE = {
    "2": ("o", style.STAR_BY_K[2][1], "subset size k = 2"),
    "3": ("^", style.STAR_BY_K[3][1], "subset size k = 3"),
    "6": ("D", style.STAR_BY_K[6][1], "subset size k = 6"),
}

#: horizontal dodge (in passes) so the three error bars at one budget
#: never overprint each other
X_DODGE = {"2": -9.0, "3": 0.0, "6": 9.0}


def load():
    """Per-dataset regret table: {ds: {k: {budget: [seed values]}}}.

    Fails loudly on any missing cell — never plots a silently partial
    sweep — and verifies the per-seed means against the frozen
    aggregated table.
    """
    data = json.loads(FROZEN.read_text())
    budgets = tuple(data["meta"]["budgets"])
    if budgets != BUDGETS:
        raise ValueError(f"unexpected pass-budget grid in frozen file: {budgets}")
    if PRODUCTION_PASSES not in budgets:
        raise ValueError("production pass count missing from the frozen sweep")

    table: dict[str, dict[str, dict[int, list[float]]]] = {}
    for ds in ORDER:
        per_seed = data["per_dataset"][ds]["per_seed"]
        if tuple(sorted(per_seed)) != SEEDS:
            raise ValueError(f"{ds}: expected seeds {SEEDS}, got {sorted(per_seed)}")
        table[ds] = {k: {b: [] for b in BUDGETS} for k in KS}
        for seed in SEEDS:
            sweep = per_seed[seed]["permutation"]
            for b in BUDGETS:
                per_k = sweep[str(b)]["per_k"]
                for k in KS:
                    table[ds][k][b].append(float(per_k[k]["mean_cheap_regret"]))
        # integrity: seed-mean must reproduce the frozen aggregated value
        agg = data["per_dataset"][ds]["aggregated"]["estimators"]
        for b in BUDGETS:
            for k in KS:
                frozen_mean = agg[f"permutation_{b}"]["per_k"][k]["mean_cheap_regret"]
                ours = statistics.mean(table[ds][k][b])
                if not math.isclose(ours, frozen_mean, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(
                        f"{ds} k={k} passes={b}: per-seed mean {ours} != "
                        f"frozen aggregated {frozen_mean}"
                    )
    return table


def build(table) -> Path:
    style.apply()
    nrows, ncols = 2, 4
    fig, axgrid = plt.subplots(nrows, ncols, figsize=(7.6, 4.0), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.30)
    axes = [ax for row in axgrid for ax in row]

    # uniform y-span across panels (hard requirement); every panel's range
    # covers all its mean +/- sd extents and the zero-regret parity line
    panel_ranges = []
    for ds in ORDER:
        ys = [0.0]
        for k in KS:
            for b in BUDGETS:
                vals = table[ds][k][b]
                m, sd = statistics.mean(vals), statistics.stdev(vals)
                ys += [m - sd, m + sd]
        panel_ranges.append((min(ys), max(ys)))
    ylims = style.uniform_yspan(panel_ranges, pad=0.04)

    for ax, ds, ylim in zip(axes, ORDER, ylims):
        ax.axhline(0.0, lw=0.8, color=style.GRAY_LIGHT, zorder=1)
        ax.axvline(PRODUCTION_PASSES, ls="--", lw=1.0, color=style.SEED_DOT,
                   zorder=1)
        for k in KS:
            marker, color, _ = K_STYLE[k]
            xs = [b + X_DODGE[k] for b in BUDGETS]
            means = [statistics.mean(table[ds][k][b]) for b in BUDGETS]
            sds = [statistics.stdev(table[ds][k][b]) for b in BUDGETS]
            ax.plot(xs, means, lw=1.0, color=color, zorder=2, alpha=0.9)
            # error bars on every marker (hard requirement)
            ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=color,
                        elinewidth=0.9, capsize=1.8, zorder=3)
            ax.scatter(xs, means, marker=marker, s=14, color=color, zorder=4)

        ax.set_xlim(0, 545)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_title(DISPLAY[ds], fontsize=9)
        ax.tick_params(labelsize=7)
        # linear axes are a hard requirement — never set log scales here
        assert ax.get_xscale() == "linear" and ax.get_yscale() == "linear"

    for i, ax in enumerate(axes[: len(ORDER)]):
        if i // ncols == nrows - 1 or i + ncols >= len(ORDER):
            ax.set_xlabel("permutation passes", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
    for r in range(nrows):
        axgrid[r][0].set_ylabel("regret vs. exact-game pick\n(validation accuracy)",
                                fontsize=8)

    # legend + protocol note in the unused 8th cell
    spare = axes[len(ORDER)]
    spare.axis("off")
    handles = [
        plt.Line2D([], [], color=K_STYLE[k][1], marker=K_STYLE[k][0],
                   markersize=4, lw=1.0, label=K_STYLE[k][2])
        for k in KS
    ]
    handles.append(plt.Line2D([], [], color=style.SEED_DOT, ls="--", lw=1.0,
                              label=f"production budget ({PRODUCTION_PASSES} passes)"))
    handles.append(plt.Line2D([], [], color=style.GRAY_LIGHT, lw=0.8,
                              label="exact-game parity (zero regret)"))
    spare.legend(handles=handles, loc="upper left", fontsize=7,
                 handlelength=1.8, borderaxespad=0.0)
    spare.text(0.0, 0.06,
               "50 estimator resamples per seed and budget;\n"
               "markers: mean across 3 training seeds,\n"
               "bars: sd across seeds. Exact game =\n"
               "2048 unique coalition evaluations.",
               transform=spare.transAxes, fontsize=6.5, va="bottom",
               color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "estimator_variance.pdf"
    # strip the timestamp so repeated builds are byte-stable
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return pdf


def main() -> None:
    pdf = build(load())
    print(f"wrote {pdf} (+.png)")


if __name__ == "__main__":
    main()
