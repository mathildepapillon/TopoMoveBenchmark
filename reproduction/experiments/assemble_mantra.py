#!/usr/bin/env python
"""Assemble the MANTRA arms into their frozen parquets.

    python experiments/assemble_mantra.py [--allow-missing]

Produces (dataset mantra_orientation, seeds 42-44):
- data/frozen/anchored3_exact_mantra.parquet  (from results/anchored3_exact)
- data/frozen/ladder_b2_mantra.parquet        (from results/ladder_b2)
- data/frozen/ladder_b2_hopse_mantra.parquet  (from results/ladder_b2_hopse)
- data/frozen/hopse_sweep_mantra.parquet      (from results/hopse_sweep_mantra)
- data/frozen/gccn_menu_mantra.parquet        (from results/gccn_menu_mantra)

Schemas match the Tier-1 counterparts (ladder/anchored: the shared ladder
columns; sweep: per-run rows; menu: per-run rows with menu_name).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DS = "mantra_orientation"
SEEDS = [42, 43, 44]

LADDER_COLUMNS = [
    "dataset", "seed", "selector", "k", "mask", "val_accuracy",
    "test_accuracy", "val_balanced_accuracy", "test_balanced_accuracy",
    "val_auroc", "test_auroc",
    "wall_seconds", "flops", "n_game_evaluations",
    "k_rung1", "k_rung2", "selected",
]


def balanced(r: dict) -> dict:
    """Balanced-accuracy fields (backfilled by mantra_balanced_eval.py).

    NaN when the cell has not been re-evaluated yet — assemble_campaign's
    audit flags any NaN here, so a parquet with missing balanced fields
    cannot silently pass as complete.
    """
    return {
        "val_balanced_accuracy":
            float(r.get("val_balanced_accuracy", math.nan)),
        "test_balanced_accuracy":
            float(r.get("test_balanced_accuracy", math.nan)),
        "val_auroc": float(r.get("val_auroc", math.nan)),
        "test_auroc": float(r.get("test_auroc", math.nan)),
    }


def ladder_rows(results, prefix, selector):
    rows, missing = [], []
    for seed in SEEDS:
        path = results / f"{prefix}_{DS}_seed{seed}.jsonl"
        if not path.exists():
            missing.append((prefix, seed))
            continue
        r = json.loads(path.read_text())
        rows.append({
            "dataset": r["dataset"], "seed": int(r["seed"]),
            "selector": selector, "k": int(r["k"]), "mask": int(r["mask"]),
            "val_accuracy": float(r["val_accuracy"]),
            "test_accuracy": float(r["test_accuracy"]),
            **balanced(r),
            "wall_seconds": float(r["stem_seconds"]) + float(r["game_seconds"])
            + float(r["continue_seconds_rung1"])
            + float(r["continue_seconds_rung2"]),
            "flops": float(r["flops"]["total_flops"]),
            "n_game_evaluations": int(r["n_game_evaluations"]),
            "k_rung1": int(r["rung1"]["k"]), "k_rung2": int(r["rung2"]["k"]),
            "selected": r["selected"],
        })
    return rows, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    all_missing = []

    # anchored k=3
    rows, missing = [], []
    for seed in SEEDS:
        path = REPO / "results" / "anchored3_exact" / \
            f"anchored3_{DS}_seed{seed}.jsonl"
        if not path.exists():
            missing.append(("anchored3", seed))
            continue
        r = json.loads(path.read_text())
        rows.append({
            "dataset": r["dataset"], "seed": int(r["seed"]),
            "selector": "anchored_k3", "k": int(r["k"]),
            "mask": int(r["mask"]),
            "val_accuracy": float(r["val_accuracy"]),
            "test_accuracy": float(r["test_accuracy"]),
            **balanced(r),
            "wall_seconds": float(r["stem_seconds"])
            + float(r["game_seconds"]) + float(r["continue_seconds"]),
            "flops": float(r["flops"]["total_flops"]),
            "n_game_evaluations": int(r["n_game_evaluations"]),
            "k_rung1": math.nan, "k_rung2": math.nan, "selected": "-",
        })
    all_missing += missing
    if rows:
        df = pd.DataFrame(rows, columns=LADDER_COLUMNS)
        out = REPO / "data" / "frozen" / "anchored3_exact_mantra.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}: {len(df)} rows")

    # GCCN + HOPSE ladders
    for results, prefix, selector, outname in [
        (REPO / "results" / "ladder_b2", "ladder_b2", "ladder_b2",
         "ladder_b2_mantra.parquet"),
        (REPO / "results" / "ladder_b2_hopse", "ladder_b2_hopse",
         "ladder_b2_hopse", "ladder_b2_hopse_mantra.parquet"),
    ]:
        rows, missing = ladder_rows(results, prefix, selector)
        all_missing += missing
        if rows:
            df = pd.DataFrame(rows, columns=LADDER_COLUMNS)
            out = REPO / "data" / "frozen" / outname
            df.to_parquet(out, index=False)
            print(f"wrote {out}: {len(df)} rows")

    # HOPSE sweep
    sys.path.insert(0, str(REPO / "experiments"))
    from run_hopse_sweep import sweep_configs
    rows, missing = [], []
    for c in sweep_configs():
        path = REPO / "results" / "hopse_sweep_mantra" / \
            f"sweep_{DS}_{c['tag']}.jsonl"
        if not path.exists():
            missing.append(("sweep", c["tag"]))
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows.append({
                "dataset": r["dataset"], "seed": int(r["seed"]),
                "selector": "hopse_sweep", "config_tag": r["config_tag"],
                "coalition": r["coalition"], "k": int(r["k"]),
                "mask": int(r["mask"]), "kind": r.get("kind"),
                "val_accuracy": float(r["val_accuracy"]),
                "test_accuracy": float(r["test_accuracy"]),
                **balanced(r),
                "train_seconds": float(r["train_seconds"]),
                "preprocessing_seconds": float(r["preprocessing_seconds"]),
                "n_epochs_trained": int(r["n_epochs_trained"]),
                "flops": float(r["flops"]["total_flops"]),
                "n_parameters_total": int(r["n_parameters_total"]),
            })
    all_missing += missing
    if rows:
        df = pd.DataFrame(rows)
        out = REPO / "data" / "frozen" / "hopse_sweep_mantra.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}: {len(df)} rows")

    # GCCN menu
    from run_gccn_menu import MENU_MASKS
    rows, missing = [], []
    for mask in MENU_MASKS:
        path = REPO / "results" / "gccn_menu_mantra" / \
            f"menu_{DS}_mask{mask}.jsonl"
        if not path.exists():
            missing.append(("menu", mask))
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows.append({
                "dataset": r["dataset"], "seed": int(r["seed"]),
                "selector": "gccn_menu", "menu_name": r["menu_name"],
                "k": int(r["k"]), "mask": int(r["mask"]),
                "val_accuracy": float(r["val_accuracy"]),
                "test_accuracy": float(r["test_accuracy"]),
                **balanced(r),
                "train_seconds": float(r["train_seconds"]),
                "n_epochs_trained": int(r["n_epochs_trained"]),
                "flops": float(r["flops"]["total_flops"]),
                "n_parameters_total": int(r["n_parameters_total"]),
            })
    all_missing += missing
    if rows:
        df = pd.DataFrame(rows)
        out = REPO / "data" / "frozen" / "gccn_menu_mantra.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}: {len(df)} rows")

    if all_missing:
        print(f"missing: {all_missing}")
        if not args.allow_missing:
            sys.exit(2)


if __name__ == "__main__":
    main()
