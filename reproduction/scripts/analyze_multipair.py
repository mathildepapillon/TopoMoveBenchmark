"""Aggregate the size-matched MANTRA multipair runs (E1+E2).

Inputs: results/mantra_attr_fig/multipair/{gccn_family,hopse}_s*.json
Outputs: results/mantra_attr_fig/multipair/summary.json + stdout tables.

Three statistics per model family:
  1. matched-sample AUROC — v_full margins as a discriminator of
     orientability WITHIN the size-matched sample (counters ~0.5)
  2. per-rank attribution profile — mean per-complex rank sums of phi
     (toward "orientable") per label, with seed sd
  3. deletion-AUC per attribution method (lower = more faithful),
     under participation removal
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
D = REPO / "results" / "mantra_attr_fig" / "multipair"

from toposhap.metrics.deletion import deletion_auc  # noqa: E402


def auroc(scores, labels):
    """Rank-based AUROC of scores for label==1."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    records = []
    for f in sorted(D.glob("*_s*.json")):
        j = json.loads(f.read_text())
        for r in j["records"]:
            r["seed"] = j["seed"]
            records.append(r)
    models = sorted({r["model"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    print(f"{len(records)} records, models {models}, seeds {seeds}\n")

    summary = {"models": {}}

    print("== matched-sample AUROC (v_full margin -> orientable) ==")
    for m in models:
        per_seed = []
        for s in seeds:
            rs = [r for r in records if r["model"] == m and r["seed"] == s]
            # margin toward NON-orientable; flip sign so score ranks
            # orientable (label 1) high
            per_seed.append(auroc([-r["v_full"] for r in rs],
                                  [r["label"] for r in rs]))
        summary["models"][m] = {"matched_auroc_per_seed": per_seed}
        print(f"  {m:10s} {np.mean(per_seed):.3f} ± {np.std(per_seed):.3f} "
              f"  per-seed {[round(a, 3) for a in per_seed]}")

    print("\n== per-rank mean sum(phi) toward 'orientable' "
          "(per complex; label 0 = non-orientable) ==")
    for m in models:
        line = {}
        for lab in (0, 1):
            sums = {0: [], 1: [], 2: []}
            for s in seeds:
                rs = [r for r in records
                      if r["model"] == m and r["seed"] == s
                      and r["label"] == lab]
                for rk in (0, 1, 2):
                    vals = []
                    for r in rs:
                        phi = -np.asarray(r["phi"])
                        ranks = np.asarray(r["player_ranks"])
                        vals.append(float(phi[ranks == rk].sum()))
                    sums[rk].append(np.mean(vals))
            line[lab] = {rk: (float(np.mean(v)), float(np.std(v)))
                         for rk, v in sums.items()}
        summary["models"][m]["rank_profile"] = line
        for lab in (0, 1):
            lb = "orient" if lab == 1 else "non-or"
            print(f"  {m:10s} {lb}: " + "  ".join(
                f"r{rk} {line[lab][rk][0]:+7.3f}±{line[lab][rk][1]:.3f}"
                for rk in (0, 1, 2)))

    print("\n== deletion-AUC (participation removal; lower = more "
          "faithful) ==")
    methods = ["toposhap", "occlusion", "gradxinput", "random"]
    for m in models:
        row = {}
        for meth in methods:
            per_seed = []
            for s in seeds:
                rs = [r for r in records
                      if r["model"] == m and r["seed"] == s
                      and meth in r["deletion_curves"]]
                if not rs:
                    continue
                aucs = []
                for r in rs:
                    # probability of the originally-predicted class:
                    # two classes, margin = logit0 - logit1, so
                    # p_pred = sigmoid(sign(v_full) * margin) in [0.5, 1]
                    # at the full coalition and bounded in [0, 1] along
                    # the curve — a well-posed deletion metric.
                    sgn = 1.0 if r["v_full"] >= 0 else -1.0
                    curve = 1.0 / (1.0 + np.exp(
                        -sgn * np.asarray(r["deletion_curves"][meth])))
                    aucs.append(deletion_auc(curve, normalize=False))
                per_seed.append(np.mean(aucs))
            if per_seed:
                row[meth] = (float(np.mean(per_seed)),
                             float(np.std(per_seed)))
        summary["models"][m]["deletion_auc"] = row
        print(f"  {m:10s} " + "  ".join(
            f"{meth} {v[0]:.3f}±{v[1]:.3f}" for meth, v in row.items()))

    (D / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote", D / "summary.json")


if __name__ == "__main__":
    main()
