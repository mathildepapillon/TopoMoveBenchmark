#!/usr/bin/env python
"""GCCN ladder-B=2 exact-FLOPs backfill: measure every rung mask locally.

    python experiments/profile_ladder_masks.py

For every distinct (dataset, mask) appearing as rung1/rung2 in
results/ladder_b2/*.jsonl at seeds {42,43,44}, plus the full mask 2047,
measure train-epoch FLOPs (forward + backward through a scalar sum of
logits) and eval-pass FLOPs on the pruned topotune_nonlinear_exact model —
FlopCounterMode, #14's conventions exactly. Writes
results/flops_local/flops_local_<dataset>.json in the same schema as #14's
flops_<dataset>.json (per_mask -> {k, train_epoch_flops, eval_pass_flops,
profile_seconds}), so data/frozen/ladder_b2_marker.parquet can be rebuilt
with zero fitted values.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from flops_profiler import (  # noqa: E402
    GCCN_COUNTER_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from run_autok import DATASETS  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.vocabulary import FULL_MASK, NEIGHBORHOODS  # noqa: E402

SEEDS = [42, 43, 44]
LADDER_RESULTS = REPO / "results" / "ladder_b2"
OUT_DIR = REPO / "results" / "flops_local"


def masks_for(dataset: str) -> list[int]:
    """Distinct rung1/rung2 masks at seeds 42-44, plus the full mask."""
    masks = {FULL_MASK}
    for seed in SEEDS:
        path = LADDER_RESULTS / f"ladder_b2_{dataset}_seed{seed}.jsonl"
        if not path.exists():
            print(f"  (missing {path.name})")
            continue
        row = json.loads(path.read_text())
        masks.add(int(row["rung1"]["mask"]))
        masks.add(int(row["rung2"]["mask"]))
    return sorted(masks)


def main() -> None:
    print(f"patches: {PATCHES}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in DATASETS:
        masks = masks_for(dataset)
        print(f"{dataset}: profiling {len(masks)} masks {masks}")
        cfg = drv.compose_config(
            dataset=dataset,
            seed=42,
            neighborhoods=list(NEIGHBORHOODS),
            output_dir=str(OUT_DIR / "runs" / dataset),
        )
        pipe = drv.build_pipeline(cfg)
        full_model = pipe.model
        train_loader = pipe.datamodule.train_dataloader()
        val_loader = pipe.datamodule.val_dataloader()

        per_mask = {}
        for mask in masks:
            model = copy.deepcopy(full_model)
            if mask != FULL_MASK:
                prune_backbone_(model.backbone.backbone, mask)
            model.to(device)
            t0 = time.perf_counter()
            train_prof = profile_train_epoch(model, train_loader, device)
            eval_prof = profile_eval_pass(model, val_loader, device)
            per_mask[str(mask)] = {
                "k": bin(mask).count("1"),
                "train_epoch_flops": train_prof["flops"],
                "eval_pass_flops": eval_prof["flops"],
                "profile_seconds": round(
                    time.perf_counter() - t0, 2
                ),
            }
            print(f"  mask {mask}: train_epoch {train_prof['flops']:.3e} "
                  f"eval_pass {eval_prof['flops']:.3e}")
            del model

        out = {
            "dataset": dataset,
            "counter": GCCN_COUNTER_NOTE,
            "per_mask": per_mask,
            "source": "local ladder_b2 backfill (rung1/rung2 masks, seeds "
                      "42-44, plus 2047); model cell/topotune_nonlinear_"
                      "exact, full 11-vocabulary bit order",
            "stem_epochs": 5,
            "prune_epoch": 5,
            "seeds": SEEDS,
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
        }
        dest = OUT_DIR / f"flops_local_{dataset}.json"
        dest.write_text(json.dumps(out, indent=1))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
