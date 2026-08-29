#!/usr/bin/env python
"""GCCN neighborhood sweep on MANTRA — the campaign's 23-config grid.

    python experiments/run_gccn_sweep.py --dataset mantra_orientation \
        --config-index 0..22

One JSONL per config (all seeds), mirroring run_hopse_sweep.py. The
grid is the campaign's frozen TopoTune-style draw, read from Stage D
(cost13.parquet, approach ``gccn_sweep``, NCI1 rows) so every dataset
sweeps the same 23 coalitions. Balanced metrics are computed in-run
from the best checkpoint (MANTRA orientability is class-imbalanced;
selection stays on plain val accuracy as everywhere else). Per-config
FLOPs are profiled in-run under the standard conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "scripts"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()  # before building any model

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from baseline_markers import TOKEN  # noqa: E402
from flops_profiler import (  # noqa: E402
    GCCN_COUNTER_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from mantra_balanced_eval import split_metrics  # noqa: E402
from resume_guard import append_partial, cell_complete, finalize, \
    load_partial  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

DATASETS = ["mantra_orientation"]
SEEDS = [42, 43, 44]
COST13 = REPO / "data" / "frozen" / "hopse13" / "results" / "cost13.parquet"

FLOPS_FORMULA = (
    "n_epochs_trained*(train_epoch + val_pass) + val_pass + test_pass; "
    "lifting preprocessing excluded, matching #14's per-run accounting"
)


def sweep_configs() -> list[dict]:
    """The campaign's frozen 23-coalition GCCN sweep grid (NCI1 draw,
    adopted verbatim for every dataset), in sorted-coalition order."""
    c13 = pd.read_parquet(COST13)
    ids = sorted(
        c13[(c13.approach == "gccn_sweep")
            & (c13.dataset == "NCI1")].subset_identity.unique()
    )
    assert len(ids) == 23, f"expected 23 sweep configs, got {len(ids)}"
    out = []
    for i, coalition in enumerate(ids):
        members = [TOKEN[t.strip()] for t in coalition.split("+")]
        out.append({
            "tag": f"cfg{i:02d}",
            "coalition": coalition,
            "members": members,
            "mask": sum(1 << NEIGHBORHOODS.index(m) for m in members),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--config-index", type=int, required=True,
                    help="0..22 into the frozen sweep grid")
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(REPO / "results" / "gccn_sweep_mantra")

    spec = sweep_configs()[args.config_index]
    tag, members = spec["tag"], spec["members"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{args.dataset}_{tag}.jsonl"
    if cell_complete(out_path, min_rows=len(SEEDS)):
        print(f"skip: {out_path} already complete")
        return
    print(f"patches: {PATCHES}")
    print(f"config {tag}: {spec['coalition']} ({len(members)} members)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resumed = {int(r["seed"]): r for r in load_partial(out_path)}
    rows = []
    profile = None
    for seed in SEEDS:
        if seed in resumed:
            row = resumed[seed]
            rows.append(row)
            if profile is None:
                profile = {k: row["flops"][k]
                           for k in ("train_epoch", "val_pass", "test_pass")}
            print(f"  seed {seed}: resumed from partial journal "
                  f"(val {row['val_accuracy']:.4f})")
            continue
        run_dir = out_dir / "runs" / f"{args.dataset}_{tag}_seed{seed}"
        cfg = drv.compose_config(
            dataset=args.dataset,
            seed=seed,
            neighborhoods=members,
            output_dir=str(run_dir),
            model=f"{drv.model_domain(args.dataset)}/{args.model}",
        )
        pipe = drv.build_pipeline(cfg)
        model = pipe.model.to(device)
        t0 = time.perf_counter()
        _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
        train_seconds = time.perf_counter() - t0
        report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)

        # balanced metrics from the tested best checkpoint (already loaded
        # into pipe.model by test_best_checkpoint)
        model = pipe.model.to(device)
        val_m = split_metrics(model, pipe.datamodule.val_dataloader(),
                              "Validation", device)
        test_m = split_metrics(model, pipe.datamodule.test_dataloader(),
                               "Test", device)

        if profile is None:  # architecture identical across seeds
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
            "selector": "gccn_sweep",
            "config_tag": tag,
            "coalition": spec["coalition"],
            "members": members,
            "mask": spec["mask"],
            "k": len(members),
            "val_accuracy": float(report["val_metrics"]["val/accuracy"]),
            "test_accuracy": float(report["test_metrics"]["test/accuracy"]),
            "val_balanced_accuracy": float(val_m["balanced"]),
            "test_balanced_accuracy": float(test_m["balanced"]),
            "val_auroc": val_m["auroc"],
            "test_auroc": test_m["auroc"],
            "balanced_eval_crosscheck": {
                "val_plain_recomputed": float(val_m["plain"]),
                "test_plain_recomputed": float(test_m["plain"]),
            },
            "train_seconds": float(train_seconds),
            "n_epochs_trained": int(epochs),
            "selected_epoch": int(report["selected_epoch"]),
            "flops": {
                "conventions": GCCN_COUNTER_NOTE,
                "formula": FLOPS_FORMULA,
                **profile,
                "total_flops": int(flops_total),
            },
            "n_parameters_total": drv.count_parameters(model),
            "model": args.model,
            "orientation": "src_to_dst",
            "patches": PATCHES,
            "config_set_note": "campaign gccn_sweep grid (cost13 NCI1 "
                               "subset_identity draw) adopted verbatim",
        })
        append_partial(out_path, rows[-1])
        print(f"  seed {seed}: val {rows[-1]['val_accuracy']:.4f} "
              f"test {rows[-1]['test_accuracy']:.4f} "
              f"(bal {rows[-1]['test_balanced_accuracy']:.4f}, "
              f"{epochs} epochs, {train_seconds:.0f}s)")

    finalize(out_path, rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
