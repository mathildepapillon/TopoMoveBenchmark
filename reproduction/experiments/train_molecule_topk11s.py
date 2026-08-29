"""Structural-lift subjects: full vocabulary, StructuralOnes features.

Higher-rank cells carry constant features (MANTRA convention), so the
lift caches no structural facts: ring-ness must be computed through
incidences. Tests whether TDL subjects trained WITHOUT the feature
shortcut become motif-aligned on the lifted GraphXAI benchmarks.

The no-selection reference point for the fair table: the record GCCN
architecture with every neighborhood, fully trained under the standard
TopoBench settings and the ground-truth-aware lifting. E3's finding is
that GEA tracks the explained model's accuracy, and the full-vocabulary
models are the most accurate family — this row bounds what TopoSHAP
gets from a stronger subject. NOTE: 49,696 params vs the param-matched
GNN baselines' 26-30k; the table must disclose this.

Usage: python train_molecule_topk11s.py DATASET
Writes results/molecule_topk11s/topk11s_{DATASET}.jsonl and checkpoints
under results/molecule_topk11s/runs/{DATASET}_s{SEED}/.
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

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

SEEDS = [42, 43, 44]


def main() -> None:
    ds = sys.argv[1]
    names = list(NEIGHBORHOODS)
    full_mask = (1 << len(names)) - 1
    out_dir = REPO / "results" / "molecule_topk11s"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"topk11s_{ds}.jsonl"
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
        run_dir = out_dir / "runs" / f"{ds}_s{seed}"
        cfg = drv.compose_config(
            dataset=ds, seed=seed, neighborhoods=names,
            output_dir=str(run_dir),
            model=(f"{drv.model_domain(ds)}/"
                   f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
            extra_overrides=[
                "transforms.graph2cell_lifting.transform_name="
                "CellCycleLiftingGT",
                "transforms.graph2cell_lifting.feature_lifting="
                "StructuralOnes",
            ],
        )
        pipe = drv.build_pipeline(cfg)
        n_params = sum(p.numel() for p in pipe.model.parameters())
        pipe.model.to(device)
        t0 = time.perf_counter()
        _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
        train_seconds = time.perf_counter() - t0
        report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
        row = {
            "dataset": ds, "seed": seed, "selector": "full_vocab",
            "k": len(names), "mask": full_mask, "members": names,
            "n_params": int(n_params),
            "val_accuracy": float(report["val_metrics"]["val/accuracy"]),
            "test_accuracy": float(report["test_metrics"]["test/accuracy"]),
            "n_epochs_trained": int(epochs),
            "selected_epoch": int(report["selected_epoch"]),
            "train_seconds": float(train_seconds),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"seed {seed}: test {row['test_accuracy']:.4f} "
              f"val {row['val_accuracy']:.4f} ({epochs} epochs, "
              f"{n_params} params)", flush=True)


if __name__ == "__main__":
    main()
