#!/usr/bin/env python
"""Exhaustive neighborhood-subset landscapes, one panel per dataset.

Appendix figure: the ground truth for the stop-rule scoring and the
out-of-sample ladder validation. Per dataset: every enumerated subset of
the 6-neighborhood vocabulary as a small gray point (3-seed mean value vs
subset size k), the best-per-k line, and the full-vocabulary point
highlighted with its across-seed sd.

Inputs (read-only): data/frozen/pooled.json
  retraining_pooled[<ds>].values_mean       -- 64 coalition means (bit i = player i)
  retraining_pooled[<ds>].v_full_per_seed   -- full-vocabulary value per seed
  retraining_pooled[<ds>].tolerance         -- frozen seed-sd summaries
  dataset_order / dataset_labels / metric_labels

Binding rules honored: linear axes, uniform y-span across panels, multi-seed
sd on the full-vocabulary marker. Per-seed per-coalition values are not in
the frozen pool (original-cluster-only), so every other marker carries the frozen
median per-coalition seed sd as a representative whisker -- see
figures/out/GAPS.md.

Deterministic: subset points are spread within each k by value rank only.

Usage: .venv/bin/python figures/build_exhaustive_landscape.py
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
OUT = REPO / "figures" / "out" / "exhaustive_landscape.pdf"

GRAY_POINT = "#9a9a9a"
BEST_LINE = "#222222"
FULL_COLOR = "#1f3d99"

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


def popcount(m: int) -> int:
    return bin(m).count("1")


def main() -> None:
    style.apply()
    pooled = json.loads(POOLED.read_text())
    retr = pooled["retraining_pooled"]
    order = [d for d in pooled["dataset_order"] if d in retr]
    labels = pooled["dataset_labels"]
    metric = pooled["metric_labels"]

    ncols, nrows = 3, 4
    assert len(order) == ncols * nrows, f"expected 12 landscapes, got {len(order)}"
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 8.4))

    # First pass: y-extent per panel (uniform span across panels afterwards).
    panel_ranges = []
    for ds in order:
        d = retr[ds]
        v = np.asarray(d["values_mean"])
        sd_full = float(np.std(d["v_full_per_seed"], ddof=1))
        lo = min(v.min(), v[63] - sd_full)
        hi = max(v.max(), v[63] + sd_full)
        panel_ranges.append((lo, hi))
    ylims = style.uniform_yspan(panel_ranges, pad=0.05)

    handles = {}
    for idx, ds in enumerate(order):
        ax = axes[idx // ncols][idx % ncols]
        d = retr[ds]
        v = np.asarray(d["values_mean"])
        sd_full = float(np.std(d["v_full_per_seed"], ddof=1))
        sd_typ = d["tolerance"]["seed_sd_median_over_coalitions"]

        # All enumerated subsets: deterministic within-k spread by value rank.
        for k in range(7):
            masks = [m for m in range(64) if popcount(m) == k]
            vals = v[masks]
            n = len(vals)
            if n == 1:
                offs = np.zeros(1)
            else:
                w = min(0.30, 0.028 * (n - 1) + 0.10)
                offs = np.linspace(-w, w, n)
            xs = k + offs[np.argsort(np.argsort(vals, kind="stable"), kind="stable")]
            h = ax.scatter(xs, vals, s=5, color=GRAY_POINT, linewidths=0, zorder=2)
            handles.setdefault("subset", h)

        # Best-per-k line (the oracle ladder the stop rule is scored against).
        best = [v[[m for m in range(64) if popcount(m) == k]].max() for k in range(7)]
        (h,) = ax.plot(range(7), best, color=BEST_LINE, lw=1.1, marker=".", ms=3.5, zorder=3)
        handles.setdefault("best", h)

        # Full-vocabulary point, real across-seed sd.
        h = ax.errorbar(
            [6], [v[63]], yerr=[sd_full], fmt="D", color=FULL_COLOR, ms=4.5,
            elinewidth=1.0, capsize=2.5, zorder=4,
        )
        handles.setdefault("full", h)

        # Representative whisker: frozen median per-coalition seed sd.
        ylo, yhi = ylims[idx]
        span = yhi - ylo
        h = ax.errorbar(
            [0.25], [ylo + 0.10 * span], yerr=[sd_typ], fmt="none",
            ecolor="#666666", elinewidth=1.0, capsize=2.5, zorder=4,
        )
        handles.setdefault("typ", h)

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-0.55, 6.55)
        ax.set_xticks(range(7))
        ax.tick_params(labelsize=7)
        name = LABEL_SHORT.get(labels[ds], labels[ds])
        ax.set_title(f"{name}\n({METRIC_SHORT[metric[ds]]})", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("subset size k (neighborhoods kept)", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("subset value", fontsize=8)

    fig.legend(
        [handles["subset"], handles["best"], handles["full"], handles["typ"]],
        [
            "one enumerated subset (3-seed mean)",
            "best subset per size k",
            "full 6-neighborhood vocabulary (sd over seeds)",
            "median per-coalition seed sd (representative)",
        ],
        loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.045),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
