"""Fill the flops column of data/frozen/ladder_b2_marker.parquet from the
post-hoc profiles in results/b2_flops_profile/ (see
experiments/profile_b2_flops.py). Idempotent; fails loudly on any
(dataset, seed) row still lacking a profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FRAME = REPO / "data" / "frozen" / "ladder_b2_marker.parquet"
PROFILES = REPO / "results" / "b2_flops_profile"


def main() -> None:
    df = pd.read_parquet(FRAME)
    totals = {}
    for p in PROFILES.glob("b2_flops_*.json"):
        r = json.loads(p.read_text())
        totals[(r["dataset"], int(r["seed"]))] = float(
            r["flops"]["total_flops"])
    missing = [
        (r.dataset, int(r.seed)) for r in df.itertuples()
        if pd.isna(r.flops) and (r.dataset, int(r.seed)) not in totals
    ]
    if missing:
        print(f"missing profiles for {missing}", file=sys.stderr)
        sys.exit(2)
    df["flops"] = [
        totals.get((r.dataset, int(r.seed)), r.flops)
        for r in df.itertuples()
    ]
    df.to_parquet(FRAME, index=False)
    print(f"patched {FRAME}: flops NaN rows now {df.flops.isna().sum()}")


if __name__ == "__main__":
    main()
