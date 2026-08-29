#!/usr/bin/env python
"""Verify the #14 marker-arm checkpoint on arrival (the 8-hour ask).

Per (dataset, k) cell, compare anchored vs plain:
  1. sd collapse — anchored's across-seed sd should fall to training-noise
     width (~0.004-0.014); predictions on record: Benzene k=2/3 bars collapse
     ~25x, Fluoride k=2/3 means +0.05.
  2. mean hold/rise — anchored's mean test accuracy must not fall below
     plain's. Any cell where it does is reported honestly and treated as the
     measurable k=2 exception; FLAG TO RESEARCHER IMMEDIATELY.

    python experiments/verify_marker_arm.py            # after ingestion
    python experiments/verify_marker_arm.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TRAINING_NOISE_SD = (0.003, 0.014)  # fixed-subset retraining noise band (#12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from toposhap.io import load_marker_arm14

    df = load_marker_arm14()
    cells = []
    for (ds, k), grp in df.groupby(["dataset", "k"]):
        plain = grp[grp.selector == "plain"].test_accuracy
        anch = grp[grp.selector == "anchored"].test_accuracy
        if plain.empty or anch.empty:
            continue
        cell = {
            "dataset": ds,
            "k": int(k),
            "plain_mean": float(plain.mean()),
            "plain_sd": float(plain.std(ddof=1)),
            "anchored_mean": float(anch.mean()),
            "anchored_sd": float(anch.std(ddof=1)),
            "n_seeds": int(min(len(plain), len(anch))),
        }
        cell["sd_collapse_factor"] = (
            cell["plain_sd"] / cell["anchored_sd"]
            if cell["anchored_sd"] > 0
            else float("inf")
        )
        cell["sd_at_training_noise"] = (
            cell["anchored_sd"] <= TRAINING_NOISE_SD[1]
        )
        cell["mean_holds"] = cell["anchored_mean"] >= cell["plain_mean"] - 1e-9
        cells.append(cell)

    if not cells:
        print("no comparable cells found — check the checkpoint schema")
        sys.exit(1)

    flags = [c for c in cells if not c["mean_holds"]]
    print(f"{len(cells)} cells; {len(flags)} anchored-below-plain flag(s)\n")
    for c in sorted(cells, key=lambda c: (c["dataset"], c["k"])):
        mark = "  " if c["mean_holds"] else "!!"
        print(
            f"{mark} {c['dataset']:>14} k={c['k']}: "
            f"anchored {c['anchored_mean']:.4f}±{c['anchored_sd']:.4f} "
            f"vs plain {c['plain_mean']:.4f}±{c['plain_sd']:.4f} "
            f"(sd collapse ×{c['sd_collapse_factor']:.1f}, "
            f"noise-level={c['sd_at_training_noise']})"
        )

    if flags:
        print(
            "\nANCHORED BELOW PLAIN in the cells above marked '!!' — show "
            "honestly in the figure, treat as the measurable k=2 exception, "
            "and flag to the researcher immediately."
        )
    if args.json:
        args.json.write_text(json.dumps(cells, indent=2))
        print(f"\nwrote {args.json}")
    sys.exit(2 if flags else 0)


if __name__ == "__main__":
    main()
