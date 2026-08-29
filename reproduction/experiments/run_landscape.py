#!/usr/bin/env python
"""Exhaustive (or size-capped) neighborhood-landscape sweep — BACKGROUND
priority: this is oracle garnish, first to be cut if compute tightens
(researcher-set). The method's own results never depend on it.

Retrains a model per subset mask and records test accuracy; masks stream to
parquet as they finish so a partial landscape is still usable.

    python experiments/run_landscape.py --dataset MUTAG --seeds 42 43 44 \
        --max-size 4          # pre-registered fallback: size<=4 exhaustive
    python experiments/run_landscape.py --dataset MUTAG --seeds 42 --full
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from toposhap.vocabulary import FULL_MASK, N_PLAYERS  # noqa: E402


def masks_by_size(max_size: int | None):
    if max_size is None:
        yield from range(1, FULL_MASK + 1)
        return
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(N_PLAYERS), size):
            yield sum(1 << i for i in combo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true",
                       help="all 2,047 subsets")
    group.add_argument("--max-size", type=int,
                       help="size-capped exhaustive (fallback: 4)")
    group.add_argument("--mask", type=int,
                       help="single subset mask (SLURM array-task mode)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mask is not None:
        masks = [args.mask]
    else:
        masks = list(masks_by_size(None if args.full else args.max_size))
    total = len(masks) * len(args.seeds)
    print(
        f"{args.dataset}: {len(masks)} masks x {len(args.seeds)} seeds = "
        f"{total} retraining runs — submit via experiments/slurm/, do not "
        "run on a login node"
    )
    # Actual retraining loop reuses TopoBenchHooks from run_recipe.py with a
    # per-mask pruned-from-init model; wired on first GPU run alongside
    # TopoBenchHooks._fit. Kept explicit for the same review-first reason.
    raise NotImplementedError(
        "wire TopoBenchHooks._fit first (experiments/run_recipe.py)"
    )


if __name__ == "__main__":
    main()
