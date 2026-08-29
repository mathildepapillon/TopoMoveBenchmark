#!/usr/bin/env python
"""GCCN menu baseline on MANTRA: the 6 canonical masks, full trainings.

    python experiments/run_gccn_menu.py --dataset mantra_orientation --mask 21
    python experiments/run_gccn_menu.py --dataset mantra_orientation \
        --mask 21 --prepare-only

Menu masks (frozen vocabulary bit order): 21=HOPSE, 26=CWN, 72=CAN,
129=CCXN, 329=SCN, 479=SCCN. One job = one (dataset, mask): a full standard
training per seed {42, 43, 44} with the model BUILT with the mask's
neighborhoods (topotune_nonlinear_exact family, orientation fixed),
best-val-checkpoint test via the normal path, measured per-run FLOPs
(FlopCounterMode, #14 conventions, profiled after all training).

Emits results/gccn_menu_mantra/menu_<ds>_mask<mask>.jsonl (one line/seed).
"""

from __future__ import annotations

import argparse
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

PATCHES = apply_all()  # before building any model

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from flops_profiler import (  # noqa: E402
    GCCN_COUNTER_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from resume_guard import (  # noqa: E402
    append_partial,
    cell_complete,
    finalize,
    load_partial,
)
from toposhap.vocabulary import mask_to_coalition  # noqa: E402

MENU_MASKS = [21, 26, 72, 129, 329, 479]
MENU_NAMES = {21: "HOPSE", 26: "CWN", 72: "CAN", 129: "CCXN", 329: "SCN",
              479: "SCCN"}
SEEDS = [42, 43, 44]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mantra_orientation")
    ap.add_argument("--mask", type=int, required=True, choices=MENU_MASKS)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "gccn_menu_mantra"))
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    members = list(mask_to_coalition(args.mask))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"menu_{args.dataset}_mask{args.mask}.jsonl"
    if not args.prepare_only and cell_complete(out_path, min_rows=len(SEEDS)):
        print(f"skip: {out_path} already complete")
        return
    print(f"patches: {PATCHES}")
    print(f"menu mask {args.mask} ({MENU_NAMES[args.mask]}): {members}")

    def compose(seed, run_dir):
        return drv.compose_config(
            dataset=args.dataset,
            seed=seed,
            neighborhoods=members,
            output_dir=str(run_dir),
            model=f"{drv.model_domain(args.dataset)}/{args.model}",
        )

    if args.prepare_only:
        cfg = compose(42, out_dir / "runs" / f"prep_mask{args.mask}")
        pipe = drv.build_pipeline(cfg, data_only=True)
        print(f"prepared {args.dataset} mask {args.mask}: "
              f"{pipe.preprocessing_seconds:.1f}s")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resumed = {int(r["seed"]): r for r in load_partial(out_path)}
    rows = []
    profile = None
    for seed in SEEDS:
        if seed in resumed:  # journaled by a previous (killed) run
            row = resumed[seed]
            rows.append(row)
            if profile is None:
                profile = {k: row["flops"][k]
                           for k in ("train_epoch", "val_pass", "test_pass")}
            print(f"  seed {seed}: resumed from partial journal "
                  f"(val {row['val_accuracy']:.4f})")
            continue
        run_dir = out_dir / "runs" / f"{args.dataset}_mask{args.mask}_s{seed}"
        cfg = compose(seed, run_dir)
        pipe = drv.build_pipeline(cfg)
        model = pipe.model.to(device)
        t0 = time.perf_counter()
        _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
        train_seconds = time.perf_counter() - t0
        report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)

        if profile is None:
            profile = {
                "train_epoch": profile_train_epoch(
                    model, pipe.datamodule.train_dataloader(), device),
                "val_pass": profile_eval_pass(
                    model, pipe.datamodule.val_dataloader(), device),
                "test_pass": profile_eval_pass(
                    model, pipe.datamodule.test_dataloader(), device,
                    "Test"),
            }
        te, vp, tp = (profile["train_epoch"]["flops"],
                      profile["val_pass"]["flops"],
                      profile["test_pass"]["flops"])
        flops_total = epochs * (te + vp) + vp + tp

        rows.append({
            "dataset": args.dataset,
            "seed": seed,
            "selector": "gccn_menu",
            "menu_name": MENU_NAMES[args.mask],
            "mask": int(args.mask),
            "k": len(members),
            "members": members,
            "val_accuracy": float(report["val_metrics"]["val/accuracy"]),
            "test_accuracy": float(report["test_metrics"]["test/accuracy"]),
            "monitored_metric":
                str(cfg.dataset.parameters.monitor_metric),
            "train_seconds": float(train_seconds),
            "n_epochs_trained": int(epochs),
            "selected_epoch": int(report["selected_epoch"]),
            "flops": {
                "conventions": GCCN_COUNTER_NOTE,
                "formula": "epochs*(train_epoch + val_pass) + val_pass + "
                           "test_pass; lifting/loader preprocessing "
                           "excluded, matching #14's per-run accounting",
                **profile,
                "total_flops": int(flops_total),
            },
            "n_parameters_total": drv.count_parameters(model),
            "n_parameters_backbone":
                drv.count_parameters(model.backbone.backbone),
            "model": args.model,
            "orientation": "src_to_dst",
            "patches": PATCHES,
        })
        append_partial(out_path, rows[-1])
        print(f"  seed {seed}: val {rows[-1]['val_accuracy']:.4f} "
              f"test {rows[-1]['test_accuracy']:.4f} ({epochs} epochs, "
              f"{train_seconds:.0f}s)")

    finalize(out_path, rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
