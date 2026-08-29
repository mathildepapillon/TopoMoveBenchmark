#!/usr/bin/env python
"""Post-hoc FLOPs profiling for the campaign's B=2 selection runs.

The original campaign predates the FLOPs profiler, so its rows in
results/ladder_b2/ carry no flops dict. FLOP counts are deterministic
given architecture and data, so this script rebuilds each recorded run's
models (full-vocabulary stem plus the two rung architectures pruned to
the recorded masks) and profiles them under the exact conventions of
run_ladder_b2.py, then writes the same flops dict with the recorded
epoch and evaluation counts:

    python experiments/profile_b2_flops.py --dataset benzene

Output: results/b2_flops_profile/b2_flops_{dataset}_seed{seed}.json,
one per seed found in results/ladder_b2/. Skips seeds already profiled.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()  # before building any model

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from flops_profiler import (  # noqa: E402
    GCCN_COUNTER_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from run_ladder_b2 import FLOPS_TOTAL_FORMULA  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    args = ap.parse_args()

    src = REPO / "results" / "ladder_b2"
    out_dir = REPO / "results" / "b2_flops_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in sorted(src.glob(f"ladder_b2_{args.dataset}_seed*.jsonl")):
        row = json.loads(path.read_text().splitlines()[0])
        if "flops" in row and row["flops"]:
            print(f"skip {path.name}: already profiled in-run")
            continue
        seed = int(row["seed"])
        out = out_dir / f"b2_flops_{args.dataset}_seed{seed}.json"
        if out.exists():
            print(f"skip {out.name}: exists")
            continue

        cfg = drv.compose_config(
            dataset=args.dataset,
            seed=seed,
            neighborhoods=list(NEIGHBORHOODS),
            output_dir=str(out_dir / "runs" / f"{args.dataset}_seed{seed}"),
            model=f"{drv.model_domain(args.dataset)}/{args.model}",
        )
        pipe = drv.build_pipeline(cfg)
        stem = pipe.model.to(device)
        train_loader = pipe.datamodule.train_dataloader()
        val_loader = pipe.datamodule.val_dataloader()
        test_loader = pipe.datamodule.test_dataloader()

        model_r1 = copy.deepcopy(stem)
        prune_backbone_(model_r1.backbone.backbone, int(row["rung1"]["mask"]))
        model_r2 = copy.deepcopy(stem)
        prune_backbone_(model_r2.backbone.backbone, int(row["rung2"]["mask"]))
        model_r1.to(device)
        model_r2.to(device)

        stem_prof = profile_train_epoch(stem, train_loader, device)
        val_prof = profile_eval_pass(stem, val_loader, device)
        test_prof = profile_eval_pass(stem, test_loader, device, "Test")
        r1_prof = profile_train_epoch(model_r1, train_loader, device)
        r2_prof = profile_train_epoch(model_r2, train_loader, device)

        e1 = int(row["n_epochs_trained_in_continuation_rung1"])
        e2 = int(row["n_epochs_trained_in_continuation_rung2"])
        stem_epochs = int(row["stem_epochs"])
        n_evals = int(row["n_game_evaluations"])
        vp, tp = val_prof["flops"], test_prof["flops"]
        total = (
            stem_epochs * (stem_prof["flops"] + vp)
            + n_evals * vp
            + e1 * (r1_prof["flops"] + vp) + vp + tp
            + e2 * (r2_prof["flops"] + vp) + vp + tp
        )
        flops = {
            "conventions": GCCN_COUNTER_NOTE,
            "total_formula": FLOPS_TOTAL_FORMULA,
            "note": "post-hoc profile of the campaign run: models rebuilt "
                    "from the recorded rung masks, counts combined with the "
                    "recorded stem/game/continuation tallies",
            "stem_train_epoch": stem_prof,
            "val_pass": val_prof,
            "test_pass": test_prof,
            "rung1_train_epoch": r1_prof,
            "rung2_train_epoch": r2_prof,
            "n_epochs_rung1": e1,
            "n_epochs_rung2": e2,
            "total_flops": int(total),
        }
        out.write_text(json.dumps(
            {"dataset": args.dataset, "seed": seed, "flops": flops}) + "\n")
        print(f"{args.dataset} seed{seed}: total {total:.3e} -> {out.name}")


if __name__ == "__main__":
    main()
