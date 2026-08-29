#!/usr/bin/env python
"""Calibration gate C: masked-game sanity vs the #14 cached benzene game.

Trains a fresh benzene stem (seed 45, 5 epochs) with the same protocol as
run_autok.py, evaluates v(2047) (and spot masks) through the
CoalitionMaskedBackbone game, and compares v(2047) against the value cached in
data/frozen/marker_arm14/results/cheap/sampled_benzene_seed45_E5.npz.
Different stem, same protocol — ballpark agreement (|delta| <= 0.05) expected,
not equality.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from toposhap.neighborhoods import CoalitionMaskedBackbone  # noqa: E402
from toposhap.vocabulary import FULL_MASK, NEIGHBORHOODS  # noqa: E402

CACHE = (
    REPO / "data" / "frozen" / "marker_arm14" / "results" / "cheap"
    / "sampled_benzene_seed45_E5.npz"
)
TOLERANCE = 0.05


def main() -> None:
    print(f"patches: {PATCHES}")
    blob = np.load(CACHE, allow_pickle=True)
    masks, values = blob["eval_masks"], blob["eval_values"]
    ref = {int(m): float(v) for m, v in zip(masks, values)}

    cfg = drv.compose_config(
        dataset="benzene",
        seed=45,
        neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(REPO / "results" / "autok" / "runs" / "gateC_stem"),
    )
    pipe = drv.build_pipeline(cfg)
    stem_seconds = drv.fit_stem(cfg, pipe, epochs=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipe.model.to(device)

    wrapper = CoalitionMaskedBackbone(model.backbone.backbone)
    val_loader = pipe.datamodule.val_dataloader()

    def v(mask: int) -> float:
        with wrapper.coalition(mask):
            acc, _ = drv.accuracy_over_loader(
                model, val_loader, "Validation", device
            )
        return acc

    print(f"stem trained in {stem_seconds:.1f}s")
    failures = 0
    for mask in (FULL_MASK, 0, 1, 132):
        if mask not in ref:
            continue
        ours = v(mask)
        delta = ours - ref[mask]
        hard = mask == FULL_MASK  # the gate; others are informational
        flag = ""
        if hard and abs(delta) > TOLERANCE:
            failures += 1
            flag = "  !! GATE C FAIL"
        print(
            f"mask {mask:5d}: ours {ours:.4f}  #14 {ref[mask]:.4f}  "
            f"delta {delta:+.4f}{flag}"
        )
    if failures:
        sys.exit(2)
    print("gate C passed: v(2047) within "
          f"{TOLERANCE} of the #14 cached game")


if __name__ == "__main__":
    main()
