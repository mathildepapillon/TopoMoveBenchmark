#!/usr/bin/env python
"""Substitute-commitment exemplar (main text, fig:substitute-commitment).

The failure mode named in appendix_selection.tex: per-seed stem reliance on
up_adjacency-0 vs. its incidence substitutes, and post-pruning viability.
Which substitute the stem commits to varies with the training seed, and the
attribution follows the commitment; the exact retraining landscape shows the
model does not need any single member of the trio.

Exemplar: Benzene, the clearest seed flip in the frozen pool -- seed 43's
stem leans on adj-0 (zero-message Shapley 2.30 vs 0.15-0.17 for the other
seeds) while seeds 42 and 44 lean on inc-1>0; the true landscape values the
best subset without any one of the trio within ~1.3 seed sd of the full set.

Inputs (read-only): data/frozen/pooled.json
  correlation["zero_message|shapley"].points[*].cheap_per_seed
      -- per-seed stem reliance (zero-message logit Shapley, valid split;
         the same cheap quantity the frozen redundancy analysis uses)
  redundancy.per_dataset.benzene.per_relation[*].best_without
      -- true value of the best subset that excludes the neighborhood
  retraining_pooled.benzene.values_mean / .v_full_per_seed / .tolerance
      -- leave-one-out values, full-set value with across-seed sd

Binding rules honored: linear axes, multi-seed, error bars on every
aggregated marker (frozen median per-coalition seed sd stands in where
per-seed coalition values are original-cluster-only -- see figures/out/GAPS.md). The
two panels carry different units (logit vs. test accuracy), so the
uniform-y-span rule binds per unit, not across the pair.

Usage: .venv/bin/python figures/build_substitute_commitment.py
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
OUT = REPO / "figures" / "out" / "substitute_commitment.pdf"

DATASET = "benzene"
TRIO = ["up_adjacency-0", "up_incidence-0", "down_incidence-1"]  # bits 0,1,2

SEED_COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}
MEAN_COLOR = "#222222"
BEST_COLOR = "#1f3d99"
FULL_BAND = "#cccccc"


def main() -> None:
    style.apply()
    pooled = json.loads(POOLED.read_text())
    players = pooled["players"]
    short = pooled["short_labels"]
    label = pooled["dataset_labels"][DATASET]

    pts = {
        p["relation"]: p
        for p in pooled["correlation"]["zero_message|shapley"]["points"]
        if p["dataset"] == DATASET
    }
    red = pooled["redundancy"]["per_dataset"][DATASET]["per_relation"]
    retr = pooled["retraining_pooled"][DATASET]
    v = np.asarray(retr["values_mean"])
    seeds = retr["seeds"]
    v_full = float(v[63])
    sd_full = float(np.std(retr["v_full_per_seed"], ddof=1))
    sd_typ = retr["tolerance"]["seed_sd_median_over_coalitions"]

    x = np.arange(len(TRIO))
    xticklabels = [short[r] for r in TRIO]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 3.0))

    # ---- Panel A: per-seed stem reliance (cheap game on the trained stem).
    cheap = np.array([pts[r]["cheap_per_seed"] for r in TRIO])  # (3 rel, 3 seed)
    for j, seed in enumerate(seeds):
        ax_a.plot(
            x, cheap[:, j], color=SEED_COLORS[seed], lw=1.0, marker="o", ms=4.5,
            mfc="none", mew=1.1, label=f"seed {seed}", zorder=3,
        )
    mean = cheap.mean(axis=1)
    sem = cheap.std(axis=1, ddof=1) / np.sqrt(cheap.shape[1])
    ax_a.errorbar(
        x + 0.13, mean, yerr=sem, fmt="D", color=MEAN_COLOR, ms=4, elinewidth=1.0,
        capsize=2.5, lw=0, label="mean over seeds (SEM)", zorder=4,
    )
    ax_a.axhline(0.0, color="#999999", lw=0.6, zorder=1)
    ax_a.annotate(
        "seed 43 commits\nto adj-0",
        xy=(0.04, cheap[0, seeds.index(43)] + 0.1), xytext=(0.42, 2.85),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"),
    )
    ax_a.annotate(
        "seeds 42, 44 lean\non inc-1>0",
        xy=(1.96, cheap[2, seeds.index(42)]), xytext=(0.95, 5.55), fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"),
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xticklabels, fontsize=8)
    ax_a.set_xlim(-0.45, 2.55)
    ax_a.set_ylim(-1.3, 6.6)
    ax_a.set_ylabel("stem-game Shapley value\n(logit units, valid split)", fontsize=8)
    ax_a.set_title(f"Stem reliance per seed — {label}", fontsize=9)
    ax_a.tick_params(labelsize=7.5)
    ax_a.legend(fontsize=7, loc="upper left")

    # ---- Panel B: post-pruning viability on the exact retraining landscape.
    loo = np.array([v[63 ^ (1 << b)] for b in range(3)])
    best_wo = np.array([red[r]["best_without"] for r in TRIO])

    ax_b.axhspan(v_full - sd_full, v_full + sd_full, color=FULL_BAND, alpha=0.5,
                 zorder=0, lw=0)
    ax_b.axhline(v_full, color="#555555", lw=1.0, ls="--", zorder=1,
                 label="full 6-neighborhood set (band: sd over seeds)")
    ax_b.errorbar(
        x - 0.09, loo, yerr=sd_typ, fmt="o", color=MEAN_COLOR, ms=4.5, mfc="none",
        mew=1.1, elinewidth=1.0, capsize=2.5, lw=0, zorder=3,
        label="removed from the full set (size-5 subset)",
    )
    ax_b.errorbar(
        x + 0.09, best_wo, yerr=sd_typ, fmt="D", color=BEST_COLOR, ms=4.5,
        elinewidth=1.0, capsize=2.5, lw=0, zorder=4,
        label="best subset without it (retrained)",
    )
    worst = v_full - best_wo.min()
    ax_b.annotate(
        f"drop any one, retrain, re-select:\nworst case −{worst:.4f} test acc.\n"
        f"(≈{worst / sd_full:.1f} seed sd)",
        xy=(2.06, best_wo[2] + 0.0006), xytext=(1.02, 0.9308), fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"),
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xticklabels, fontsize=8)
    ax_b.set_xlim(-0.45, 2.55)
    ax_b.set_ylim(0.8975, 0.9350)
    ax_b.set_ylabel("test accuracy (exact landscape)", fontsize=8)
    ax_b.set_title(f"Post-pruning viability — {label}", fontsize=9)
    ax_b.tick_params(labelsize=7.5)
    ax_b.legend(fontsize=6.6, loc="lower center")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})
    print(f"wrote {OUT}")
    print(f"  stem reliance per seed (logit): "
          f"{ {short[r]: [round(c, 3) for c in pts[r]['cheap_per_seed']] for r in TRIO} }")
    print(f"  viability: full={v_full:.4f}±{sd_full:.4f}, "
          f"LOO={np.round(loo, 4).tolist()}, best_without={np.round(best_wo, 4).tolist()}")


if __name__ == "__main__":
    main()
