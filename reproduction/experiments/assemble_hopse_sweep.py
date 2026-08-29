#!/usr/bin/env python
"""Assemble the Tier-1 HOPSE sweep into data/frozen/hopse_sweep_t1.parquet.

    python experiments/assemble_hopse_sweep.py [--allow-missing]

One row per (dataset, config, seed); 2 datasets x 12 configs x 3 seeds = 72.
flops = measured per-run model total (preprocessing FLOPs excluded — no
frozen measured record for Tier-1 datasets; preprocessing_seconds clocked,
charged to the first seed per config, mirroring cost13's convention).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "hopse_sweep_t1"
CITATIONS_RESULTS = REPO / "results" / "hopse_sweep_citations"
CITATIONS_OUT = REPO / "data" / "frozen" / "hopse_sweep_citations.parquet"
CITATION_DATASETS = ["cocitation_cora", "cocitation_citeseer"]
OUT = REPO / "data" / "frozen" / "hopse_sweep_t1.parquet"
DATASETS = ["MUTAG", "NCI109", "alkane_carbonyl"]

COLUMNS = [
    "dataset",
    "seed",
    "selector",
    "config_tag",
    "coalition",
    "k",
    "mask",
    "kind",
    "val_accuracy",
    "test_accuracy",
    "train_seconds",
    "preprocessing_seconds",
    "n_epochs_trained",
    "flops",
    "n_parameters_total",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--citations", action="store_true",
                    help="assemble the node-level citation batch instead "
                         "(results/hopse_sweep_citations -> "
                         "data/frozen/hopse_sweep_citations.parquet)")
    args = ap.parse_args()
    results_dir = CITATIONS_RESULTS if args.citations else RESULTS
    out = CITATIONS_OUT if args.citations else OUT
    datasets = CITATION_DATASETS if args.citations else DATASETS

    sys.path.insert(0, str(REPO / "experiments"))
    from run_hopse_sweep import sweep_configs

    tags = [c["tag"] for c in sweep_configs()]
    rows, missing = [], []
    for dataset in datasets:
        for tag in tags:
            path = results_dir / f"sweep_{dataset}_{tag}.jsonl"
            if not path.exists():
                missing.append((dataset, tag))
                continue
            with open(path) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows.append({
                        "dataset": r["dataset"],
                        "seed": int(r["seed"]),
                        "selector": "hopse_sweep",
                        "config_tag": r["config_tag"],
                        "coalition": r["coalition"],
                        "k": int(r["k"]),
                        "mask": int(r["mask"]),
                        "kind": r.get("kind"),
                        "val_accuracy": float(r["val_accuracy"]),
                        "test_accuracy": float(r["test_accuracy"]),
                        "train_seconds": float(r["train_seconds"]),
                        "preprocessing_seconds":
                            float(r["preprocessing_seconds"]),
                        "n_epochs_trained": int(r["n_epochs_trained"]),
                        "flops": float(r["flops"]["total_flops"]),
                        "n_parameters_total": int(r["n_parameters_total"]),
                    })

    if missing:
        print(f"missing {len(missing)} configs: {missing}")
        if not args.allow_missing:
            sys.exit(2)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["dataset", "config_tag", "seed"]
    ).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    print(df.groupby(["dataset"]).agg(
        n=("test_accuracy", "size"),
        best_val_selected_test=("test_accuracy", "max"),
    ).to_string())


if __name__ == "__main__":
    main()
