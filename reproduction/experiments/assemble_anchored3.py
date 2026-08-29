#!/usr/bin/env python
"""Assemble anchored k=3 exact rows into data/frozen/anchored3_exact.parquet.

    python experiments/assemble_anchored3.py [--allow-missing]

Columns match the ladder parquets (selector='anchored_k3'; k_rung1/k_rung2
NaN and selected='-' since this arm has no rungs); flops = the measured
per-cell total; wall_seconds = stem + game + continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "anchored3_exact"
OUT = REPO / "data" / "frozen" / "anchored3_exact.parquet"

DATASETS = [
    "benzene",
    "fluoride_carbonyl",
    "mutagenicity_gxai",
    "NCI1",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
]
SEEDS = [42, 43, 44]

COLUMNS = [
    "dataset",
    "seed",
    "selector",
    "k",
    "mask",
    "val_accuracy",
    "test_accuracy",
    "wall_seconds",
    "flops",
    "n_game_evaluations",
    "k_rung1",
    "k_rung2",
    "selected",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--t1", action="store_true",
                    help="assemble the Tier-1 coverage datasets -> "
                         "anchored3_exact_t1.parquet")
    args = ap.parse_args()

    if args.t1:
        datasets = ["MUTAG", "NCI109", "alkane_carbonyl"]
        out = REPO / "data" / "frozen" / "anchored3_exact_t1.parquet"
    else:
        datasets = DATASETS
        out = OUT

    rows, missing = [], []
    for dataset in datasets:
        for seed in SEEDS:
            path = RESULTS / f"anchored3_{dataset}_seed{seed}.jsonl"
            if not path.exists():
                missing.append((dataset, seed))
                continue
            r = json.loads(path.read_text())
            rows.append({
                "dataset": r["dataset"],
                "seed": int(r["seed"]),
                "selector": "anchored_k3",
                "k": int(r["k"]),
                "mask": int(r["mask"]),
                "val_accuracy": float(r["val_accuracy"]),
                "test_accuracy": float(r["test_accuracy"]),
                "wall_seconds": float(r["stem_seconds"])
                + float(r["game_seconds"])
                + float(r["continue_seconds"]),
                "flops": float(r["flops"]["total_flops"]),
                "n_game_evaluations": int(r["n_game_evaluations"]),
                "k_rung1": math.nan,
                "k_rung2": math.nan,
                "selected": "-",
            })

    if missing:
        print(f"missing {len(missing)} cells: {missing}")
        if not args.allow_missing:
            sys.exit(2)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["dataset", "seed"]
    ).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    print(df.to_string())


if __name__ == "__main__":
    main()
