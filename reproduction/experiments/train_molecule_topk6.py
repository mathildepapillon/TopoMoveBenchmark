"""Arm-1 subjects at k=6: top-6-by-phi GCCNs, fully trained.

The protocol's safe default is k=6 (top-k neighborhoods by the stem
game's Shapley values). The per-seed phi is already journaled in
results/anchored3_exact/anchored3_{ds}_seed{seed}.jsonl, so no new game
is played: derive the top-6 mask, compose the full vocabulary under the
ground-truth-aware lifting, prune to the mask, and run the standard
full training (default epochs, early stopping, best-val checkpoint).

Usage: python train_molecule_topk6.py DATASET
Writes results/molecule_topk6/topk6_{DATASET}.jsonl (one row per seed)
and checkpoints under results/molecule_topk6/runs/{DATASET}_s{SEED}/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import os

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")
os.environ.setdefault(
    "TOPOSHAP_GRAPHXAI_DATA",
    str(REPO / "external" / "TopoBench" / "datasets" / "graphxai"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

SEEDS = [42, 43, 44]
K = 6


def main() -> None:
    ds = sys.argv[1]
    names = list(NEIGHBORHOODS)
    out_dir = REPO / "results" / "molecule_topk6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"topk6_{ds}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line:
                r = json.loads(line)
                done[r["seed"]] = r

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        if seed in done:
            print(f"seed {seed}: already trained "
                  f"(test {done[seed]['test_accuracy']:.4f})", flush=True)
            continue
        stem = json.loads(
            (REPO / "results" / "anchored3_exact"
             / f"anchored3_{ds}_seed{seed}.jsonl").read_text())
        phi = np.asarray(stem["phi"])
        assert len(phi) == len(names)
        top = np.argsort(-phi)[:K]
        members = [names[i] for i in sorted(top)]
        mask = int(sum(1 << int(i) for i in top))
        print(f"seed {seed}: top-{K} by phi -> mask {mask} {members}",
              flush=True)

        run_dir = out_dir / "runs" / f"{ds}_s{seed}"
        cfg = drv.compose_config(
            dataset=ds, seed=seed, neighborhoods=names,
            output_dir=str(run_dir),
            model=(f"{drv.model_domain(ds)}/"
                   f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
            extra_overrides=[
                "transforms.graph2cell_lifting.transform_name="
                "CellCycleLiftingGT",
            ],
        )
        pipe = drv.build_pipeline(cfg)
        prune_backbone_(pipe.model.backbone.backbone, mask)
        pipe.model.to(device)
        t0 = time.perf_counter()
        _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
        train_seconds = time.perf_counter() - t0
        report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
        row = {
            "dataset": ds, "seed": seed, "selector": "topk_phi", "k": K,
            "mask": mask, "members": members,
            "val_accuracy": float(report["val_metrics"]["val/accuracy"]),
            "test_accuracy": float(report["test_metrics"]["test/accuracy"]),
            "n_epochs_trained": int(epochs),
            "selected_epoch": int(report["selected_epoch"]),
            "train_seconds": float(train_seconds),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"seed {seed}: test {row['test_accuracy']:.4f} "
              f"val {row['val_accuracy']:.4f} ({epochs} epochs)", flush=True)


if __name__ == "__main__":
    main()
