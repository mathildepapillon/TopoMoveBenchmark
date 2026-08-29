#!/usr/bin/env python
"""Per-seed retraining-game Shapley traces, one panel per dataset.

Appendix figure extending the main-text substitute-commitment figure
(figures/build_substitute_commitment.py) to all datasets with an exact
landscape: for every dataset, the exact retraining-game Shapley value of
each of the 6 neighborhoods, per training seed (thin traces) and pooled
(mean with SEM). The adjacency/incidence substitute trio from the
main-text figure is shaded.

Inputs (read-only): data/frozen/pooled.json
  retraining_pooled[<ds>].shapley_per_seed  -- exact Shapley per player per seed
  retraining_pooled[<ds>].shapley_mean / .shapley_sem
  retraining_pooled[<ds>].seeds
  players / short_labels / dataset_order / dataset_labels / metric_labels

Binding rules honored: linear axes, uniform y-span across panels, per-seed
data drawn wherever seeds exist, SEM bars on the aggregated (mean) marker.

Usage: .venv/bin/python figures/build_seed_traces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POOLED = REPO / "data" / "frozen" / "pooled.json"
OUT = REPO / "figures" / "out" / "seed_traces.pdf"

# Okabe-Ito, colorblind-safe, fixed assignment seed -> hue.
SEED_COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}
MEAN_COLOR = "#222222"
TRIO_SHADE = "#dcdcdc"

METRIC_SHORT = {
    "test accuracy": "test acc.",
    "negative test MAE": "neg. test MAE",
    "test node accuracy": "node acc.",
}

# Display-only shortening so adjacent panel titles cannot collide.
LABEL_SHORT = {
    "ZINC-subset (fixed recipe)": "ZINC (fixed recipe)",
    "ZINC-subset (strong)": "ZINC (strong)",
}


def main() -> None:
    style.apply()
    pooled = json.loads(POOLED.read_text())
    retr = pooled["retraining_pooled"]
    order = [d for d in pooled["dataset_order"] if d in retr]
    labels = pooled["dataset_labels"]
    metric = pooled["metric_labels"]
    players = pooled["players"]
    short = pooled["short_labels"]

    ncols, nrows = 3, 4
    assert len(order) == ncols * nrows, f"expected 12 landscapes, got {len(order)}"
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 8.6))

    # Panel y-extents (per-seed values plus mean +/- SEM), then uniform span.
    panel_ranges = []
    for ds in order:
        d = retr[ds]
        per_seed = np.array([d["shapley_per_seed"][pl] for pl in players])
        mean = np.array([d["shapley_mean"][pl] for pl in players])
        sem = np.array([d["shapley_sem"][pl] for pl in players])
        lo = min(per_seed.min(), (mean - sem).min(), 0.0)
        hi = max(per_seed.max(), (mean + sem).max(), 0.0)
        panel_ranges.append((lo, hi))
    ylims = style.uniform_yspan(panel_ranges, pad=0.05)

    x = np.arange(len(players))
    handles, leg_labels = [], []
    for idx, ds in enumerate(order):
        ax = axes[idx // ncols][idx % ncols]
        d = retr[ds]
        seeds = d["seeds"]
        per_seed = np.array([d["shapley_per_seed"][pl] for pl in players])
        mean = np.array([d["shapley_mean"][pl] for pl in players])
        sem = np.array([d["shapley_sem"][pl] for pl in players])

        # Substitute trio shading (adj-0 and its two incidence substitutes).
        hs = ax.axvspan(-0.45, 2.45, color=TRIO_SHADE, alpha=0.45, zorder=0, lw=0)
        ax.axhline(0.0, color="#999999", lw=0.6, zorder=1)

        for j, seed in enumerate(seeds):
            (h,) = ax.plot(
                x, per_seed[:, j], color=SEED_COLORS[seed], lw=0.9, alpha=0.9,
                marker="o", ms=3.0, mfc="none", mew=0.9, zorder=3,
            )
            if idx == 0:
                handles.append(h)
                leg_labels.append(f"seed {seed}")
        hm = ax.errorbar(
            x + 0.22, mean, yerr=sem, fmt="D", color=MEAN_COLOR, ms=3.2,
            elinewidth=0.9, capsize=2.0, lw=0, zorder=4,
        )
        if idx == 0:
            handles.append(hm)
            leg_labels.append("mean over seeds (SEM)")
            handles.append(hs)
            leg_labels.append("adjacency + incidence substitute trio")

        ax.set_ylim(*ylims[idx])
        ax.set_xlim(-0.6, len(players) - 0.25)
        ax.set_xticks(x)
        if idx // ncols == nrows - 1:
            ax.set_xticklabels([short[pl] for pl in players], rotation=45,
                               ha="right", fontsize=6.5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=7)
        name = LABEL_SHORT.get(labels[ds], labels[ds])
        ax.set_title(f"{name}\n({METRIC_SHORT[metric[ds]]})", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("retraining Shapley value", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("neighborhood", fontsize=8)

    fig.legend(handles, leg_labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
