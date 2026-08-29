#!/usr/bin/env python
"""Assemble the auto-k JSONL rows into a frozen parquet.

    python experiments/assemble_autok.py [--allow-missing]            # forward
    python experiments/assemble_autok.py --backward [--allow-missing]

Forward: results/autok/autok_*.jsonl -> data/frozen/autok_real.parquet
(selector autok_greedy). Backward: results/autok_backward/
autok_backward_*.jsonl -> data/frozen/autok_backward.parquet (selector
autok_backward). One row per (dataset, seed); wall_seconds = stem + game +
continue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

DATASETS = [
    "benzene",
    "fluoride_carbonyl",
    "mutagenicity_gxai",
    "NCI1",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
    # B=1 backfill (2026-08-28): tier-1 datasets, seeds 42-44 only
    "MUTAG",
    "NCI109",
    "alkane_carbonyl",
    "mantra_orientation",
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
    "n_game_evaluations",
    "n_epochs_trained_in_continuation",
    "v_full",
    "binomial_se",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true",
                    help="assemble whatever rows exist instead of requiring "
                         "all 35 cells")
    ap.add_argument("--backward", action="store_true",
                    help="assemble the backward-elimination arm instead")
    args = ap.parse_args()

    if args.backward:
        results = REPO / "results" / "autok_backward"
        prefix, selector = "autok_backward", "autok_backward"
        out = REPO / "data" / "frozen" / "autok_backward.parquet"
    else:
        results = REPO / "results" / "autok"
        prefix, selector = "autok", "autok_greedy"
        out = REPO / "data" / "frozen" / "autok_real.parquet"

    rows, missing = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
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
                        + float(r["continue_seconds"]),
                        "n_game_evaluations": int(r["n_game_evaluations"]),
                        "n_epochs_trained_in_continuation": int(
                            r["n_epochs_trained_in_continuation"]
                        ),
                        "v_full": float(r["v_full"]),
                        "binomial_se": float(r["binomial_se"]),
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
