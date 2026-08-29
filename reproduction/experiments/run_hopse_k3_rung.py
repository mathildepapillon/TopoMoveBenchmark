#!/usr/bin/env python
"""Fixed-size k=3 reference in the HOPSE family: the ladder's k=3 rung.

    python experiments/run_hopse_k3_rung.py --dataset benzene --seed 42

NOT a new selection rule: the coalition is the size-3 rung of the
backward-elimination ladder the cell's HOPSE ladder-B=2 run already built
(read from its recorded JSONL trace — every full elimination visits k=3
exactly once, no choices). This arm warm-continues that rung exactly like
rung1/rung2 and prices it with the shared stem+walk cost of its cell.

Label discipline: 'fixed size k=3 (ladder rung)' — NOT anchored (no anchor
prior exists in HOPSE space).

Stems were not checkpointed by the ladder cells, so the stem is RETRAINED
here with the identical protocol (same seed, 5 epochs) and the retrained
stem is priced instead — noted per row (stem_retrained=true). The walk
(game) cost is carried over from the cell's record: its n_game_evaluations
priced at this run's measured val-pass FLOPs, and its recorded game_seconds
included in wall_seconds.

Emits results/hopse_k3_rung/k3_<ds>_seed<seed>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()  # before building any model

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from flops_profiler import (  # noqa: E402
    HOPSE_CONVENTIONS_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from resume_guard import cell_complete  # noqa: E402
from hopse_masking import HopseCoalitionMasker  # noqa: E402
from run_ladder_b2_hopse import (  # noqa: E402
    DATASETS,
    hopse_compose,
    mask_members,
)

LADDER_RESULTS = REPO / "results" / "ladder_b2_hopse"

FLOPS_FORMULA = (
    "stem_epochs*(stem_train_epoch + val_pass) [retrained stem priced] + "
    "cell_n_game_evaluations*val_pass [walk reused from the cell's record] "
    "+ epochs*(k3_train_epoch + val_pass) + val_pass + test_pass; "
    "preprocessing excluded as in the cell's own accounting"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "hopse_k3_rung"))
    args = ap.parse_args()

    final_path = (Path(args.out_dir)
                  / f"k3_{args.dataset}_seed{args.seed}.jsonl")
    if cell_complete(final_path):
        print(f"skip: {final_path} already complete")
        return

    cell_path = (LADDER_RESULTS
                 / f"ladder_b2_hopse_{args.dataset}_seed{args.seed}.jsonl")
    cell = json.loads(cell_path.read_text())
    k3 = next(r for r in cell["ladder"] if r["k"] == 3)
    mask = int(k3["mask"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"patches: {PATCHES}")
    print(f"k=3 rung from cell trace: mask={mask} "
          f"members={mask_members(mask)} cheap_v={k3['cheap_v']:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # retrained stem (identical protocol; cells kept no stem checkpoints)
    stem_cfg = hopse_compose(args.dataset, args.seed, run_dir / "stem")
    pipe = drv.build_pipeline(stem_cfg)
    t0 = time.perf_counter()
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    # fix the k=3 rung's encoding mask, continue warm, test best-val ckpt
    masker = HopseCoalitionMasker(model.feature_encoder)
    masker.fix(mask)
    cont_cfg = hopse_compose(args.dataset, args.seed, run_dir / "continue")
    t0 = time.perf_counter()
    _, ckpt_cb, _, epochs = drv.fit_continuation(cont_cfg, pipe)
    continue_seconds = time.perf_counter() - t0
    report = drv.test_best_checkpoint(cont_cfg, pipe, ckpt_cb, device)
    val_accuracy = float(report["val_metrics"]["val/accuracy"])
    test_accuracy = float(report["test_metrics"]["test/accuracy"])

    # measured FLOPs after all training (mask does not change HOPSE dispatch)
    train_loader = pipe.datamodule.train_dataloader()
    train_prof = profile_train_epoch(model, train_loader, device)
    val_prof = profile_eval_pass(
        model, pipe.datamodule.val_dataloader(), device)
    test_prof = profile_eval_pass(
        model, pipe.datamodule.test_dataloader(), device, "Test")

    walk_evals = int(cell["n_game_evaluations"])
    vp, tp = val_prof["flops"], test_prof["flops"]
    flops_total = (
        args.stem_epochs * (train_prof["flops"] + vp)
        + walk_evals * vp
        + epochs * (train_prof["flops"] + vp) + vp + tp
    )
    wall = stem_seconds + float(cell["game_seconds"]) + continue_seconds

    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selector": "hopse_k3_rung",
        "label": "fixed size k=3 (ladder rung) — NOT anchored; no anchor "
                 "prior exists in HOPSE space",
        "mask": mask,
        "k": 3,
        "members": mask_members(mask),
        "cheap_v": float(k3["cheap_v"]),
        "source_cell": str(cell_path.name),
        "stem_retrained": True,
        "stem_retrained_note": "ladder cells kept no stem checkpoints; "
                               "stem retrained with the identical protocol "
                               "and priced here",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "monitored_metric":
            str(stem_cfg.dataset.parameters.monitor_metric),
        "stem_seconds": float(stem_seconds),
        "cell_game_seconds": float(cell["game_seconds"]),
        "continue_seconds": float(continue_seconds),
        "wall_seconds": float(wall),
        "n_epochs": int(epochs),
        "selected_epoch": int(report["selected_epoch"]),
        "flops": {
            "conventions": HOPSE_CONVENTIONS_NOTE,
            "formula": FLOPS_FORMULA,
            "train_epoch": train_prof,
            "val_pass": val_prof,
            "test_pass": test_prof,
            "cell_n_game_evaluations": walk_evals,
            "total_flops": int(flops_total),
        },
        "n_parameters_total": drv.count_parameters(model),
        "stem_epochs": int(args.stem_epochs),
        "model": "hopse_m",
        "orientation": "src_to_dst",
        "patches": PATCHES,
    }
    out_path = out_dir / f"k3_{args.dataset}_seed{args.seed}.jsonl"
    out_path.write_text(json.dumps(row) + "\n")
    print(f"wrote {out_path}")
    print(f"k=3 mask={mask} val={val_accuracy:.4f} test={test_accuracy:.4f} "
          f"wall={wall:.1f}s flops={flops_total:.3e}")


if __name__ == "__main__":
    main()
