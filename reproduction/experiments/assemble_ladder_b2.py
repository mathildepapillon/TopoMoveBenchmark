#!/usr/bin/env python
"""Assemble ladder B=2 rows into a frozen parquet.

    python experiments/assemble_ladder_b2.py [--allow-missing]           # GCCN
    python experiments/assemble_ladder_b2.py --hopse [--allow-missing]

GCCN: results/ladder_b2 -> data/frozen/ladder_b2_marker.parquet (selector
ladder_b2, 35 cells). HOPSE: results/ladder_b2_hopse ->
data/frozen/ladder_b2_hopse.parquet (selector ladder_b2_hopse, 20 molecule
cells). One row per (dataset, seed): the validation-selected rung's
k/mask/val/test; wall_seconds = stem + game + BOTH continuations (the honest
B=2 price); flops left NaN (assembled downstream).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "ladder_b2"
OUT = REPO / "data" / "frozen" / "ladder_b2_marker.parquet"

DATASETS = [
    "benzene",
    "fluoride_carbonyl",
    "mutagenicity_gxai",
    "NCI1",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
]
SEEDS = [42, 43, 44, 45, 46]

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
    ap.add_argument("--hopse", action="store_true",
                    help="assemble the HOPSE arm (4 molecule datasets)")
    ap.add_argument("--t1", action="store_true",
                    help="assemble the Tier-1 coverage datasets instead")
    args = ap.parse_args()

    if args.hopse and args.t1:
        results = REPO / "results" / "ladder_b2_hopse"
        prefix, selector = "ladder_b2_hopse", "ladder_b2_hopse"
        out = REPO / "data" / "frozen" / "ladder_b2_hopse_t1.parquet"
        datasets = ["MUTAG", "NCI109", "alkane_carbonyl"]
        seeds = [42, 43, 44]
    elif args.hopse:
        results = REPO / "results" / "ladder_b2_hopse"
        prefix, selector = "ladder_b2_hopse", "ladder_b2_hopse"
        out = REPO / "data" / "frozen" / "ladder_b2_hopse.parquet"
        # Scope decision (researcher, 2026-08-29): the HOPSE family is
        # evaluated on complex-level tasks only. Node-level citation arms
        # were run (raw JSONLs under results/ladder_b2_hopse/) but are
        # excluded from the paper's frames for a consistent story.
        datasets = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai",
                    "NCI1"]
        seeds = [42, 43, 44]  # amended protocol: match the sweep baselines
    elif args.t1:
        results = RESULTS
        prefix, selector = "ladder_b2", "ladder_b2"
        out = REPO / "data" / "frozen" / "ladder_b2_t1.parquet"
        datasets = ["MUTAG", "NCI109", "alkane_carbonyl"]
        seeds = [42, 43, 44]
    else:
        results = RESULTS
        prefix, selector = "ladder_b2", "ladder_b2"
        out = OUT
        datasets = DATASETS
        seeds = SEEDS

    rows, missing = [], []
    for dataset in datasets:
        for seed in seeds:
            path = results / f"{prefix}_{dataset}_seed{seed}.jsonl"
            if not path.exists():
                missing.append((dataset, seed))
                continue
            with open(path) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows.append({
                        "dataset": r["dataset"],
                        "seed": int(r["seed"]),
                        "selector": selector,
                        "k": int(r["k"]),
                        "mask": int(r["mask"]),
                        "val_accuracy": float(r["val_accuracy"]),
                        "test_accuracy": float(r["test_accuracy"]),
                        "wall_seconds": float(r["stem_seconds"])
                        + float(r["game_seconds"])
                        + float(r["continue_seconds_rung1"])
                        + float(r["continue_seconds_rung2"]),
                        "flops": float(
                            r["flops"]["total_flops"]
                        ) if "flops" in r else math.nan,
                        "n_game_evaluations": int(r["n_game_evaluations"]),
                        "k_rung1": int(r["rung1"]["k"]),
                        "k_rung2": int(r["rung2"]["k"]),
                        "selected": r["selected"],
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
