"""Arm-1 baseline subjects: GCN and GIN on the GraphXAI molecule datasets.

The fairness rule for Table 1 is same-split, same-metric, same-population.
The frozen baseline suite (data/frozen/baselines2) used its own
stratified-shuffle split, so its numbers cannot sit next to ours; the
graph explainers must run on GNNs trained under the SAME TopoBench splits
as the cell-domain subjects. This script trains those GNNs.

Same pipeline as the cell subjects: identical dataset config and split
params, standard full training (default max epochs, early stopping
patience 25, best-val checkpoint). Before training, it hard-asserts that
the graph-domain test split contains exactly the same molecules
(orig_index sets, full and gt-positive) as the cell-domain pipeline used
by experiments/molecule_participation_gea.py — a split mismatch aborts.

Usage: python train_graph_gnns.py DATASET [SEED ...]
Writes results/graph_baselines/train_{DATASET}.jsonl (one row per
arch x seed) and checkpoints under results/graph_baselines/runs/.
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
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

ARCHS = ["gcn", "gin"]
DEFAULT_SEEDS = [42, 43, 44]

#: Parameter-matched to the k=6 GCCN subjects (28,716 params, uniform
#: across datasets): GCN 29,804 (+3.8%), GIN 26,220 (-8.7%). The
#: TopoBench per-dataset graph defaults are far smaller (2.7k-9.6k) and
#: would invite a hobbled-baseline objection; 3 layers also matches the
#: GraphXAI paper's GNN depth.
ARCH_SIZES = {"gcn": (96, 3), "gin": (64, 3)}


def compose_graph(ds: str, seed: int, arch: str, out_dir: Path):
    hidden, layers = ARCH_SIZES[arch]
    return drv.compose_config(
        dataset=ds, seed=seed, neighborhoods=[],
        output_dir=str(out_dir),
        model=f"graph/{arch}",
        neighborhood_key=None,
        extra_overrides=[
            f"model.feature_encoder.out_channels={hidden}",
            f"model.backbone.hidden_channels={hidden}",
            f"model.backbone.num_layers={layers}",
        ],
    )


def test_index_sets(pipe) -> tuple[set[int], set[int]]:
    """(all test orig_index, gt-positive test orig_index) for a pipeline."""
    all_idx, pos_idx = set(), set()
    for d in pipe.datamodule.dataset_test.data_lst:
        orig = int(np.asarray(d.orig_index).reshape(-1)[0])
        all_idx.add(orig)
        if bool(np.asarray(d.gt_positive).reshape(-1)[0]):
            pos_idx.add(orig)
    return all_idx, pos_idx


def assert_split_matches_cell(ds: str, seed: int, graph_pipe) -> int:
    """The graph test split must equal the cell pipeline's, molecule for
    molecule. Returns the gt-positive population size."""
    cell_cfg = drv.compose_config(
        dataset=ds, seed=seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(REPO / "results" / "graph_baselines"
                       / f"cfg_splitcheck_{ds}_s{seed}"),
        model=(f"{drv.model_domain(ds)}/"
               f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
        extra_overrides=[
            "transforms.graph2cell_lifting.transform_name="
            "CellCycleLiftingGT",
        ],
    )
    cell_pipe = drv.build_pipeline(cell_cfg, data_only=True)
    g_all, g_pos = test_index_sets(graph_pipe)
    c_all, c_pos = test_index_sets(cell_pipe)
    if g_all != c_all or g_pos != c_pos:
        raise SystemExit(
            f"SPLIT MISMATCH {ds} seed {seed}: graph test "
            f"{len(g_all)} ({len(g_pos)} gt+) vs cell test "
            f"{len(c_all)} ({len(c_pos)} gt+); "
            f"sym-diff {len(g_all ^ c_all)} — refusing to train")
    print(f"split check ok: {len(g_all)} test molecules, "
          f"{len(g_pos)} gt-positive, identical to the cell pipeline",
          flush=True)
    return len(g_pos)


def main() -> None:
    ds = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or DEFAULT_SEEDS
    out_dir = REPO / "results" / "graph_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_{ds}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line:
                r = json.loads(line)
                done.add((r["arch"], r["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checked_seeds: dict[int, int] = {}
    for arch in ARCHS:
        for seed in seeds:
            if (arch, seed) in done:
                print(f"{arch} s{seed}: already trained", flush=True)
                continue
            run_dir = out_dir / "runs" / f"{ds}_{arch}_s{seed}"
            cfg = compose_graph(ds, seed, arch, run_dir)
            pipe = drv.build_pipeline(cfg)
            if seed not in checked_seeds:
                checked_seeds[seed] = assert_split_matches_cell(
                    ds, seed, pipe)
            n_params = sum(p.numel() for p in pipe.model.parameters())
            pipe.model.to(device)
            t0 = time.perf_counter()
            _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
            train_seconds = time.perf_counter() - t0
            report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
            row = {
                "dataset": ds, "arch": arch, "seed": seed,
                "n_params": int(n_params),
                "n_test_gt_positive": checked_seeds[seed],
                "val_accuracy": float(
                    report["val_metrics"]["val/accuracy"]),
                "test_accuracy": float(
                    report["test_metrics"]["test/accuracy"]),
                "n_epochs_trained": int(epochs),
                "selected_epoch": int(report["selected_epoch"]),
                "train_seconds": float(train_seconds),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{arch} s{seed}: test {row['test_accuracy']:.4f} "
                  f"val {row['val_accuracy']:.4f} ({epochs} epochs, "
                  f"{n_params} params)", flush=True)


if __name__ == "__main__":
    main()
