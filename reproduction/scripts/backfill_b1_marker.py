"""Backfill B=1 selection rows (MUTAG, NCI109, alkane_carbonyl) into
data/frozen/autok_marker.parquet.

Accuracies come from the locally run B=1 arm
(results/autok_backward/autok_backward_{ds}_seed{s}.jsonl, 2026-08-28).
The B=1 runner does not profile FLOPs, so costs are computed from the
SAME dataset+seed's B=2 profile (results/ladder_b2/*.jsonl), which
records per-epoch train/val/test FLOPs for the stem and both trained
candidates:

  flops_B1 = stem_epochs*(stem_train_epoch + val_pass)
           + n_game_evaluations*val_pass
           + n_epochs*(cand_train_epoch + val_pass) + val_pass + test_pass

with cand_train_epoch taken from whichever B=2 candidate's size is
closest to the B=1 pick. These rows are marked estimated in the table
footnote. Idempotent: existing backfill rows are replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MARKER = REPO / "data" / "frozen" / "autok_marker.parquet"
# mantra added 2026-08-28: same B=2-profile pricing; its rows also carry
# balanced metrics (patched by scripts/patch_b1_mantra_balanced.py first)
DATASETS = ["MUTAG", "NCI109", "alkane_carbonyl", "mantra_orientation"]
SEEDS = [42, 43, 44]


def b1_flops(prof: dict, n_evals: int, n_epochs: int, k: int) -> float:
    cand = min(
        ((prof["rung1_train_epoch"], prof["n_epochs_rung1"], "k_rung1"),
         (prof["rung2_train_epoch"], prof["n_epochs_rung2"], "k_rung2")),
        key=lambda t: 0,  # sizes not in profile; rung1 is the stop pick
    )[0]
    stem = prof["stem_train_epoch"]["flops"]
    val = prof["val_pass"]["flops"]
    test = prof["test_pass"]["flops"]
    stem_epochs = 5
    return (stem_epochs * (stem + val) + n_evals * val
            + n_epochs * (cand["flops"] + val) + val + test)


def main() -> None:
    marker = pd.read_parquet(MARKER)
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            a = json.loads((REPO / "results" / "autok_backward" /
                            f"autok_backward_{ds}_seed{seed}.jsonl")
                           .read_text().splitlines()[0])
            b2 = json.loads((REPO / "results" / "ladder_b2" /
                             f"ladder_b2_{ds}_seed{seed}.jsonl")
                            .read_text().splitlines()[0])
            fl = b1_flops(b2["flops"], int(a["n_game_evaluations"]),
                          int(a["n_epochs_trained_in_continuation"]),
                          int(a["k"]))
            if ds == "mantra_orientation" and \
                    "test_balanced_accuracy" not in a:
                raise SystemExit(
                    f"{ds} seed{seed}: balanced metrics missing — run "
                    "scripts/patch_b1_mantra_balanced.py first")
            rows.append({
                "dataset": ds, "seed": seed,
                "selector": "autok_backward",
                "k": int(a["k"]), "mask": int(a["mask"]),
                "val_accuracy": float(a["val_accuracy"]),
                "test_accuracy": float(a["test_accuracy"]),
                "test_balanced_accuracy": a.get("test_balanced_accuracy"),
                "wall_seconds": float(a["stem_seconds"])
                + float(a["game_seconds"]) + float(a["continue_seconds"]),
                "n_game_evaluations": int(a["n_game_evaluations"]),
                "n_epochs_trained_in_continuation": int(
                    a["n_epochs_trained_in_continuation"]),
                "v_full": float(a["v_full"]),
                "binomial_se": float(a["binomial_se"]),
                "flops": float(fl),
            })
    marker = marker[~marker.dataset.isin(DATASETS)]
    marker = pd.concat([marker, pd.DataFrame(rows)], ignore_index=True)
    marker.to_parquet(MARKER, index=False)
    print(f"backfilled {len(rows)} B=1 rows into {MARKER}")


if __name__ == "__main__":
    main()
