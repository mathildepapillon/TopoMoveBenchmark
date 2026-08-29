#!/usr/bin/env python
"""Tier-1 HOPSE sweep baseline: #13's 12 configs on a new TUDataset dataset.

    python experiments/run_hopse_sweep.py --dataset MUTAG --config-index 0
    python experiments/run_hopse_sweep.py --dataset MUTAG --config-index 0 \
        --prepare-only          # CPU pre-pass: config-specific encodings

One job = one (dataset, config): preprocess-if-cold (clocked), then a full
standard HOPSE training per seed {42, 43, 44} (cell/hopse_m built with ONLY
the config's neighborhoods, exactly like #13's sweep), best-val-checkpoint
test via the normal path, and a measured-FLOPs profile per config (train
epoch, val pass, test pass — FlopCounterMode, profiled after all training).

The 12 configs: cost13.parquet's hopse_sweep subset_identity values are
per-dataset random draws in #13 (39 distinct across its 4 datasets; only
the shipped default cfg00 and the full menu cfg08 are shared), so there is
no single canonical 12-set for NEW datasets. This arm adopts NCI1's frozen
12-config set (the sibling TUDataset draw, which contains both anchors) for
every Tier-1 dataset — the same 12 configs for MUTAG and NCI109. Config
membership is read from analysis13.json (tags cfg00..cfg11, cross-checked
against cost13.parquet's subset_identity strings).

Emits results/hopse_sweep_t1/sweep_<ds>_<tag>.jsonl (one line per seed).
Preprocessing convention mirrors cost13: the cold preprocessing cost is
charged once per (dataset, config) and recorded on the first seed's row.
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
    HOPSE_CONVENTIONS_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from resume_guard import (  # noqa: E402
    append_partial,
    cell_complete,
    finalize,
    load_partial,
)
from hopse_masking import HOPSE_NEIGHBORHOODS  # noqa: E402

DATASETS = ["MUTAG", "NCI109", "mantra_orientation", "alkane_carbonyl",
            # citation fill (2026-08-29): node-level, encodings cached
            "cocitation_cora", "cocitation_citeseer"]
SEEDS = [42, 43, 44]
ANALYSIS13 = REPO / "data" / "frozen" / "hopse13" / "results" / "analysis13.json"

FLOPS_FORMULA = (
    "per run: epochs*(train_epoch + val_pass) + val_pass + test_pass; "
    "encoding-preprocessing FLOPs have no frozen measured record for Tier-1 "
    "datasets and are excluded (preprocessing_seconds clocked instead)"
)


def sweep_configs() -> list[dict]:
    """#13's NCI1 12-config set (tag, members, coalition), frozen order."""
    cfgs = json.load(open(ANALYSIS13))["per_dataset"]["NCI1"]["sweep"][
        "configs"
    ]
    out = []
    for c in cfgs:
        out.append({
            "tag": c["tag"],
            "members": list(c["members"]),
            "coalition": c["coalition"],
            "kind": c.get("kind"),
        })
    assert len(out) == 12
    return out


def members_mask(members: list[str]) -> int:
    """Bitmask over the 8-player HOPSE vocabulary (analysis13 bit order)."""
    mask = 0
    for m in members:
        mask |= 1 << HOPSE_NEIGHBORHOODS.index(m)
    return mask


def compose(dataset, seed, members, output_dir):
    return drv.compose_config(
        dataset=dataset,
        seed=seed,
        neighborhoods=members,
        output_dir=str(output_dir),
        model=f"{drv.model_domain(dataset)}/hopse_m",
        neighborhood_key="model.preprocessing_params.neighborhoods",
        extra_overrides=["transforms.hopse_encoding.device=cpu"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--config-index", type=int, required=True,
                    help="0..11 into #13's NCI1 sweep config set")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(
            REPO / "results"
            / ("hopse_sweep_mantra" if args.dataset == "mantra_orientation"
               else "hopse_sweep_citations"
               if args.dataset.startswith("cocitation")
               else "hopse_sweep_t1")
        )

    cfg_spec = sweep_configs()[args.config_index]
    tag, members = cfg_spec["tag"], cfg_spec["members"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{args.dataset}_{tag}.jsonl"
    if not args.prepare_only and cell_complete(out_path, min_rows=len(SEEDS)):
        print(f"skip: {out_path} already complete")
        return
    print(f"patches: {PATCHES}")
    print(f"config {tag}: {cfg_spec['coalition']} ({len(members)} members)")

    if args.prepare_only:
        cfg = compose(args.dataset, 42, members,
                      out_dir / "runs" / f"prep_{args.dataset}_{tag}")
        pipe = drv.build_pipeline(cfg, data_only=True)
        prep = {"dataset": args.dataset, "tag": tag,
                "preprocessing_seconds": pipe.preprocessing_seconds}
        (out_dir / f"prep_{args.dataset}_{tag}.json").write_text(
            json.dumps(prep, indent=1))
        print(f"prepared {args.dataset}/{tag}: "
              f"{pipe.preprocessing_seconds:.1f}s (recorded)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep_path = out_dir / f"prep_{args.dataset}_{tag}.json"
    prep_seconds = (
        json.loads(prep_path.read_text())["preprocessing_seconds"]
        if prep_path.exists() else None
    )

    resumed = {int(r["seed"]): r for r in load_partial(out_path)}
    rows = []
    profile = None
    for i, seed in enumerate(SEEDS):
        if seed in resumed:  # journaled by a previous (killed) run
            row = resumed[seed]
            rows.append(row)
            if profile is None:
                profile = {k: row["flops"][k]
                           for k in ("train_epoch", "val_pass", "test_pass")}
            print(f"  seed {seed}: resumed from partial journal "
                  f"(val {row['val_accuracy']:.4f})")
            continue
        run_dir = out_dir / "runs" / f"{args.dataset}_{tag}_seed{seed}"
        cfg = compose(args.dataset, seed, members, run_dir)
        pipe = drv.build_pipeline(cfg)
        model = pipe.model.to(device)
        t0 = time.perf_counter()
        _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
        train_seconds = time.perf_counter() - t0
        report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)

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
            "selector": "hopse_sweep",
            "config_tag": tag,
            "coalition": cfg_spec["coalition"],
            "members": members,
            "mask": members_mask(members),
            "k": len(members),
            "kind": cfg_spec["kind"],
            "val_accuracy": float(report["val_metrics"]["val/accuracy"]),
            "test_accuracy": float(report["test_metrics"]["test/accuracy"]),
            "train_seconds": float(train_seconds),
            "preprocessing_seconds":
                float(prep_seconds) if (i == 0 and prep_seconds) else 0.0,
            "n_epochs_trained": int(epochs),
            "selected_epoch": int(report["selected_epoch"]),
            "flops": {
                "conventions": HOPSE_CONVENTIONS_NOTE,
                "formula": FLOPS_FORMULA,
                **profile,
                "total_flops": int(flops_total),
            },
            "n_parameters_total": drv.count_parameters(model),
            "model": "hopse_m",
            "orientation": "src_to_dst",
            "patches": PATCHES,
            "config_set_note": "NCI1's frozen 12-config #13 draw adopted "
                               "for Tier-1 datasets (per-dataset draws "
                               "differ in #13; only cfg00/cfg08 shared)",
        })
        append_partial(out_path, rows[-1])
        print(f"  seed {seed}: val {rows[-1]['val_accuracy']:.4f} "
              f"test {rows[-1]['test_accuracy']:.4f} ({epochs} epochs, "
              f"{train_seconds:.0f}s, flops {flops_total:.3e})")

    finalize(out_path, rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
