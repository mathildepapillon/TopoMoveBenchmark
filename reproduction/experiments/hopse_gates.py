#!/usr/bin/env python
"""HOPSE arm calibration gate A: full-model reference + zeroing mechanism.

For each molecule dataset and matched seeds {42, 43, 44}, train the full
8-block HOPSE model (cell/hopse_m, standard trainer settings) with the
encoding-block masker installed at the FULL mask (a no-op if the zeroing is
wired right), evaluate the best-val checkpoint on test via the normal path,
and compare the per-dataset test mean against #13's full_menu reference in
data/frozen/hopse13/results/analysis13.json (tolerance 0.02).

Mechanism checks per dataset (on the seed-42 model): masked val at the full
mask must equal the unmasked val exactly, and dropping block 0 must change
the val accuracy (the zeroing must bite).

Exit 2 on any gate failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from hopse_masking import HOPSE_FULL_MASK, HopseCoalitionMasker  # noqa: E402
from run_ladder_b2_hopse import hopse_compose  # noqa: E402

DATASETS = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai", "NCI1"]
SEEDS = [42, 43, 44]
TOLERANCE = 0.02
ANALYSIS13 = REPO / "data" / "frozen" / "hopse13" / "results" / "analysis13.json"


def main() -> None:
    print(f"patches: {PATCHES}")
    refs = json.load(open(ANALYSIS13))["per_dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    out = {}

    for dataset in DATASETS:
        ref = refs[dataset]["references"]["full_menu"]["test"]["mean"]
        tests = []
        for seed in SEEDS:
            run_dir = (REPO / "results" / "ladder_b2_hopse" / "runs"
                       / f"gateA_{dataset}_seed{seed}")
            cfg = hopse_compose(dataset, seed, run_dir)
            pipe = drv.build_pipeline(cfg)
            model = pipe.model.to(device)
            masker = HopseCoalitionMasker(model.feature_encoder)
            masker.fix(HOPSE_FULL_MASK)  # no-op if zeroing is wired right
            _, ckpt_cb, seconds, epochs = drv.fit_continuation(cfg, pipe)
            report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
            test = float(report["test_metrics"]["test/accuracy"])
            tests.append(test)
            print(f"  {dataset} s{seed}: test {test:.4f} "
                  f"({epochs} epochs, {seconds:.0f}s)")

            if seed == 42:  # mechanism checks on the trained model
                plain_val, _ = drv.evaluate_split(
                    model, pipe.datamodule, "val", device
                )
                with masker.coalition(HOPSE_FULL_MASK):
                    full_val, _ = drv.evaluate_split(
                        model, pipe.datamodule, "val", device
                    )
                with masker.coalition(HOPSE_FULL_MASK & ~1):
                    drop0_val, _ = drv.evaluate_split(
                        model, pipe.datamodule, "val", device
                    )
                noop_ok = abs(full_val - plain_val) < 1e-12
                bites_ok = abs(drop0_val - plain_val) > 1e-12
                print(f"  {dataset} mechanism: full-mask no-op "
                      f"{'OK' if noop_ok else 'FAIL'} "
                      f"(plain {plain_val:.4f} masked {full_val:.4f}); "
                      f"drop-block-0 bites "
                      f"{'OK' if bites_ok else 'FAIL'} "
                      f"(v {drop0_val:.4f})")
                if not noop_ok or not bites_ok:
                    failures += 1
            masker.restore()

        mean = sum(tests) / len(tests)
        delta = mean - ref
        ok = abs(delta) <= TOLERANCE
        if not ok:
            failures += 1
        out[dataset] = {"tests": tests, "mean": mean, "ref": ref,
                        "delta": delta, "ok": ok}
        print(f"GATE A {dataset}: mean {mean:.4f} vs #13 full_menu "
              f"{ref:.4f} (delta {delta:+.4f}) "
              f"{'PASS' if ok else '!! FAIL'}")

    dest = REPO / "results" / "ladder_b2_hopse" / "gateA.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    if failures:
        sys.exit(2)
    print("gate A passed on all datasets")


if __name__ == "__main__":
    main()
