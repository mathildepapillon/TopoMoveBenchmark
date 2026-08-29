#!/usr/bin/env python
"""Build sweep-best baseline markers from Stage D (cost13.parquet).

Three baselines, one row per (dataset, baseline, seed):
  hopse_best      — HOPSE's own 12-config sweep (4 molecule datasets)
  gccn_sweep_best — the GCCN neighborhood sweep, TopoTune-style (~23 cfg, 7 ds)
  originals_best  — TopoBench's original TDL architectures as-is (can/cccn/ccxn/cwn)

Conventions (matching the menu marker):
  * selection: per-seed validation-best config; test of that config/seed.
  * cost: running the grid once = sum over configs of the per-config mean
    cost (this is what FINDING the best charges; hard budget-matching rule).
  * FLOPs: measured where Stage D profiled them (hopse, originals);
    for gccn_sweep (#12-sourced, unprofiled) train FLOPs are assembled as
    train_epoch_flops(mask) x per-dataset mean epochs from the #14 continued
    runs — an ESTIMATE, disclosed; wall-clock is exact for all three.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
FROZEN = REPO / "data" / "frozen"

from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

TOKEN = {
    "adj-0": "up_adjacency-0", "adj-1": "up_adjacency-1",
    "coadj-1": "down_adjacency-1", "coadj-2": "down_adjacency-2",
    "inc-0>1": "up_incidence-0", "inc-1>2": "up_incidence-1",
    "inc-1>0": "down_incidence-1", "inc-2>1": "down_incidence-2",
    "2-adj-0": "2-up_adjacency-0", "adj2-0": "2-up_adjacency-0",
    "inc-0>2": "2-up_incidence-0", "2-inc-0>2": "2-up_incidence-0",
    "inc-2>0": "2-down_incidence-2", "2-inc-2>0": "2-down_incidence-2",
}
BIT = {n: i for i, n in enumerate(NEIGHBORHOODS)}


def coalition_to_mask(s: str) -> int | None:
    mask = 0
    for tok in s.split("+"):
        name = TOKEN.get(tok.strip())
        if name is None:
            return None
        mask |= 1 << BIT[name]
    return mask


def per_seed_val_best(sub: pd.DataFrame) -> list[dict]:
    rows = []
    for seed, g in sub.groupby("seed"):
        best = g.loc[g.val_accuracy.idxmax()]
        rows.append(dict(seed=int(seed),
                         test_accuracy=float(best.test_accuracy),
                         val_accuracy=float(best.val_accuracy),
                         winner=str(best.get("subset_identity",
                                              best.get("architecture", "")))))
    return rows


def main() -> None:
    df = pd.read_parquet(FROZEN / "hopse13" / "results" / "cost13.parquet")

    # per-dataset mean continuation epochs from #14 (for the gccn estimate)
    epochs_ds: dict[str, float] = {}
    jdir = FROZEN / "marker_arm14" / "results" / "continued"
    acc: dict[str, list] = {}
    for f in jdir.glob("continued5_*.jsonl"):
        for line in f.read_text().strip().splitlines():
            r = json.loads(line)
            acc.setdefault(r["dataset"], []).append(
                r["n_epochs_trained_in_continuation"]
            )
    epochs_ds = {ds: float(np.mean(v)) for ds, v in acc.items()}

    out = []
    for baseline, approach in (("hopse_best", "hopse_sweep"),
                               ("gccn_sweep_best", "gccn_sweep"),
                               ("originals_best", "topobench_original")):
        sub_all = df[df.approach == approach]
        key = ("architecture" if approach == "topobench_original"
               else "subset_identity")
        for ds, sub in sub_all.groupby("dataset"):
            n_cfg = sub[key].nunique()
            wall = float(sub.groupby(key).total_seconds.mean().sum())
            if sub.total_flops.notna().all():
                flops = float(sub.groupby(key).total_flops.mean().sum())
                flops_estimated = False
            else:
                prof_p = (FROZEN / "marker_arm14" / "results" / "flops"
                          / f"flops_{ds}.json")
                prof = json.loads(prof_p.read_text())["per_mask"]
                masks_p = np.array([int(m) for m in prof])
                tf = np.array([prof[str(m)]["train_epoch_flops"]
                               for m in masks_p], dtype=float)
                X = np.column_stack([np.ones(len(masks_p))]
                                    + [(masks_p >> i) & 1 for i in range(11)]
                                    ).astype(float)
                coef, *_ = np.linalg.lstsq(X, tf, rcond=None)

                def tef(mask: int) -> float:
                    if str(mask) in prof:
                        return prof[str(mask)]["train_epoch_flops"]
                    x = np.array([1.0] + [(mask >> i) & 1 for i in range(11)])
                    return float(x @ coef)

                total, unknown = 0.0, 0
                for ident in sub[key].unique():
                    mask = coalition_to_mask(ident)
                    if mask is None:
                        unknown += 1
                        continue
                    total += tef(mask) * epochs_ds.get(ds, float("nan"))
                if unknown:
                    print(f"  WARNING {ds}/{approach}: {unknown} configs with "
                          "unparsed coalition tokens skipped in FLOPs sum")
                flops = total
                flops_estimated = True
            for row in per_seed_val_best(sub):
                out.append(dict(dataset=ds, baseline=baseline, n_configs=n_cfg,
                                flops=flops, wall_seconds=wall,
                                flops_estimated=flops_estimated, **row))
    frame = pd.DataFrame(out)
    dest = FROZEN / "baseline_markers.parquet"
    frame.to_parquet(dest, index=False)
    print(f"wrote {dest} ({len(frame)} rows)")
    print(frame.groupby(['dataset', 'baseline']).agg(
        test=('test_accuracy', 'mean'), sd=('test_accuracy', 'std'),
        flops=('flops', 'first'), n_cfg=('n_configs', 'first')).to_string())


if __name__ == "__main__":
    main()
