#!/usr/bin/env python
"""Tier-1 gate (c): HOPSE full-mask no-op exactness on a new dataset.

    python experiments/hopse_noop_gate.py --dataset MUTAG

Trains a 5-epoch full-8-block HOPSE stem, then checks (i) masked validation
accuracy at the full mask equals the unmasked validation accuracy exactly,
and (ii) dropping block 0 changes it (the zeroing must bite). Exit 2 on
failure.
"""

from __future__ import annotations

import argparse
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
from run_ladder_b2_hopse import DATASETS, hopse_compose  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"patches: {PATCHES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = (REPO / "results" / "ladder_b2_hopse" / "runs"
               / f"noopgate_{args.dataset}_seed{args.seed}")
    cfg = hopse_compose(args.dataset, args.seed, run_dir)
    pipe = drv.build_pipeline(cfg)
    drv.fit_stem(cfg, pipe, epochs=5)
    model = pipe.model.to(device)

    plain, _ = drv.evaluate_split(model, pipe.datamodule, "val", device)
    masker = HopseCoalitionMasker(model.feature_encoder)
    with masker.coalition(HOPSE_FULL_MASK):
        full, _ = drv.evaluate_split(model, pipe.datamodule, "val", device)

    def val_logits(mask: int) -> torch.Tensor:
        """Concatenated val-set logits under a coalition (fixed order)."""
        model.eval()
        model.state_str = "Validation"
        outs = []
        with torch.no_grad(), masker.coalition(mask):
            for batch in pipe.datamodule.val_dataloader():
                batch = batch.to(device)
                batch["model_state"] = "Validation"
                outs.append(model.forward(batch)["logits"].detach().cpu())
        return torch.cat(outs)

    # no-op: full-mask logits must equal unmasked logits exactly; bites:
    # dropping block 0 must move the LOGITS (accuracy is too coarse a probe
    # on tiny validation sets — e.g. MUTAG's ~47 graphs).
    logits_plain = val_logits(HOPSE_FULL_MASK)  # masker no-op at full
    masker.restore()
    model.eval()
    model.state_str = "Validation"
    outs = []
    with torch.no_grad():
        for batch in pipe.datamodule.val_dataloader():
            batch = batch.to(device)
            batch["model_state"] = "Validation"
            outs.append(model.forward(batch)["logits"].detach().cpu())
    logits_unmasked = torch.cat(outs)
    masker2 = HopseCoalitionMasker(model.feature_encoder)

    def val_logits2(mask):
        model.eval()
        model.state_str = "Validation"
        outs = []
        with torch.no_grad(), masker2.coalition(mask):
            for batch in pipe.datamodule.val_dataloader():
                batch = batch.to(device)
                batch["model_state"] = "Validation"
                outs.append(model.forward(batch)["logits"].detach().cpu())
        return torch.cat(outs)

    logits_drop0 = val_logits2(HOPSE_FULL_MASK & ~1)
    noop_logit_gap = float((logits_plain - logits_unmasked).abs().max())
    bite_logit_gap = float((logits_drop0 - logits_unmasked).abs().max())

    # At the full mask the masker short-circuits (no zeroing code runs), so
    # the only residual is GPU kernel run-to-run nondeterminism between two
    # separate passes (~1e-7 observed on H200). Thresholds: no-op gap must
    # sit at that noise floor (<= 1e-5) with exact accuracy equality; the
    # drop-block-0 gap must be a real signal (> 1e-3 and >> the noise).
    noop_ok = abs(full - plain) < 1e-12 and noop_logit_gap <= 1e-5
    bites_ok = bite_logit_gap > 1e-3 and bite_logit_gap > 100 * max(
        noop_logit_gap, 1e-12
    )
    print(f"{args.dataset}: plain acc {plain:.4f} masked-full acc "
          f"{full:.4f}; no-op logit gap {noop_logit_gap:.3e}; drop-block-0 "
          f"logit gap {bite_logit_gap:.3e} | no-op "
          f"{'OK' if noop_ok else 'FAIL'} bites "
          f"{'OK' if bites_ok else 'FAIL'}")
    if not (noop_ok and bites_ok):
        sys.exit(2)
    print("gate (c) passed")


if __name__ == "__main__":
    main()
