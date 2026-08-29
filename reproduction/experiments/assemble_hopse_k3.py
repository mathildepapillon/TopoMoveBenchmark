#!/usr/bin/env python
"""Assemble the HOPSE fixed-size k=3 (ladder rung) arm.

    python experiments/assemble_hopse_k3.py [--allow-missing]

-> data/frozen/hopse_k3_rung.parquet with columns [dataset, seed,
selector='hopse_k3_rung', k=3, mask, val_accuracy, test_accuracy,
val_balanced_accuracy, test_balanced_accuracy (NaN where no balanced
backfill exists), wall_seconds, flops, n_epochs] over every HOPSE-family
cell that has a ladder record (4 molecule datasets + MUTAG + NCI109 +
MANTRA, seeds 42-44).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "hopse_k3_rung"
OUT = REPO / "data" / "frozen" / "hopse_k3_rung.parquet"

DATASETS = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai", "NCI1",
            "MUTAG", "NCI109", "mantra_orientation"]
SEEDS = [42, 43, 44]

COLUMNS = ["dataset", "seed", "selector", "k", "mask", "val_accuracy",
           "test_accuracy", "val_balanced_accuracy",
           "test_balanced_accuracy", "val_auroc", "test_auroc",
           "wall_seconds", "flops", "n_epochs"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    rows, missing = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = RESULTS / f"k3_{dataset}_seed{seed}.jsonl"
            if not path.exists():
                missing.append((dataset, seed))
                continue
            r = json.loads(path.read_text())
            rows.append({
                "dataset": r["dataset"], "seed": int(r["seed"]),
                "selector": "hopse_k3_rung", "k": 3,
                "mask": int(r["mask"]),
                "val_accuracy": float(r["val_accuracy"]),
                "test_accuracy": float(r["test_accuracy"]),
                # balanced-accuracy backfill (mantra_balanced_eval.py) is
                # add-only in the JSONLs; carry it when present, NaN else
                "val_balanced_accuracy": float(
                    r.get("val_balanced_accuracy", float("nan"))),
                "test_balanced_accuracy": float(
                    r.get("test_balanced_accuracy", float("nan"))),
                "val_auroc": float(r.get("val_auroc", float("nan"))),
                "test_auroc": float(r.get("test_auroc", float("nan"))),
                "wall_seconds": float(r["wall_seconds"]),
                "flops": float(r["flops"]["total_flops"]),
                "n_epochs": int(r["n_epochs"]),
            })

    if missing:
        print(f"missing {len(missing)} cells: {missing}")
        if not args.allow_missing:
            sys.exit(2)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["dataset", "seed"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows")
    print(df.to_string())


if __name__ == "__main__":
    main()
