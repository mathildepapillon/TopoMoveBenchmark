"""Enumerate matched-count cross-label pairs in the MANTRA test set.

A matched pair is (orientable, non-orientable) test complexes with
identical cell counts (n0, n1, n2) — size cannot separate them, so any
model signal on the pair is topological. Writes
results/mantra_attr_fig/matched_pairs.json: every matched count-triple,
one deterministic pair per triple, plus a capped sample for the
multi-pair attribution runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import os

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import R, gccn_compose  # noqa: E402

CAP = 25


def main() -> None:
    pipe = drv.build_pipeline(gccn_compose(42, "matchedpairs"))
    loader = pipe.datamodule.test_dataloader()
    ds = loader.dataset

    by_counts: dict[tuple, dict[int, list[int]]] = {}
    labels = {}
    for idx in range(len(ds.data_lst)):
        d = ds.data_lst[idx]
        counts = (int(d.x_0.shape[0]), int(d.x_1.shape[0]),
                  int(d.x_2.shape[0]))
        y = int(d.y)
        labels[idx] = y
        by_counts.setdefault(counts, {}).setdefault(y, []).append(idx)

    pairs = []
    for counts, groups in sorted(by_counts.items()):
        if len(groups) < 2:
            continue
        a, b = sorted(groups.keys())
        pairs.append({
            "counts": list(counts),
            "n_cells": sum(counts),
            f"label_{a}": groups[a][0],
            f"label_{b}": groups[b][0],
            "group_sizes": {str(k): len(v) for k, v in groups.items()},
        })

    pairs.sort(key=lambda p: p["n_cells"])
    # deterministic per-triple complex sample: up to K per label,
    # smallest test indices first
    K = 8
    for p_ in pairs:
        counts = tuple(p_["counts"])
        p_["sampled_indices"] = {
            str(y): sorted(idxs)[:K]
            for y, idxs in by_counts[counts].items()}
    sample = pairs[:CAP]
    out = {
        "n_test": len(ds.data_lst),
        "n_matched_triples": len(pairs),
        "cap": CAP,
        "note": "label 1 = orientable; pair = first index per label per "
                "count-triple; sample = smallest n_cells first",
        "pairs_all": pairs,
        "sample": sample,
    }
    path = R / "mantra_attr_fig" / "matched_pairs.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"test={out['n_test']} matched triples={len(pairs)} "
          f"sample={len(sample)} "
          f"cells range {sample[0]['n_cells']}-{sample[-1]['n_cells']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
