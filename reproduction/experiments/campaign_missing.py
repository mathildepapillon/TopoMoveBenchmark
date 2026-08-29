#!/usr/bin/env python
"""Missing-cell inventory for the MANTRA + k3-rung campaign.

    python experiments/campaign_missing.py            # two csv lines
    python experiments/campaign_missing.py --verbose  # human inventory

Prints exactly two lines (possibly empty):
  line 1: missing mantra_arrays.sbatch array indices (csv)
  line 2: missing hopse_k3_array.sbatch array indices (csv)

Completeness uses the same test the runners use to skip (parseable JSONL
with the expected row count), so a resubmission of these indices only runs
what is genuinely missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from resume_guard import cell_complete  # noqa: E402

DS = "mantra_orientation"
SEEDS = [42, 43, 44]
MENU = [21, 26, 72, 129, 329, 479]
SWEEP_TAGS = [  # frozen order = run_hopse_sweep.sweep_configs()
    "cfg00_mask193", "cfg01_mask195", "cfg02_mask32", "cfg03_mask249",
    "cfg04_mask60", "cfg05_mask130", "cfg06_mask150", "cfg07_mask128",
    "cfg08_mask255", "cfg09_mask164", "cfg10_mask243", "cfg11_mask94",
]
K3_DATASETS = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai", "NCI1",
               "MUTAG", "NCI109", "mantra_orientation"]

R = REPO / "results"


def mantra_cells() -> list[tuple[int, Path, int]]:
    """(array index, expected output, expected rows) for mantra_arrays."""
    cells = []
    for i, seed in enumerate(SEEDS):
        cells.append((i, R / "ladder_b2" / f"ladder_b2_{DS}_seed{seed}.jsonl",
                      1))
        cells.append((3 + i, R / "anchored3_exact" /
                      f"anchored3_{DS}_seed{seed}.jsonl", 1))
        cells.append((6 + i, R / "ladder_b2_hopse" /
                      f"ladder_b2_hopse_{DS}_seed{seed}.jsonl", 1))
    for c, tag in enumerate(SWEEP_TAGS):
        cells.append((9 + c, R / "hopse_sweep_mantra" /
                      f"sweep_{DS}_{tag}.jsonl", 3))
    for m, mask in enumerate(MENU):
        cells.append((21 + m, R / "gccn_menu_mantra" /
                      f"menu_{DS}_mask{mask}.jsonl", 3))
    return sorted(cells)


def k3_cells() -> list[tuple[int, Path, int]]:
    """(array index, expected output, expected rows) for hopse_k3_array."""
    return [
        (d * 3 + s, R / "hopse_k3_rung" / f"k3_{ds}_seed{seed}.jsonl", 1)
        for d, ds in enumerate(K3_DATASETS)
        for s, seed in enumerate(SEEDS)
    ]


def missing(cells) -> list[int]:
    return [i for i, path, rows in cells
            if not cell_complete(path, min_rows=rows)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    m_mantra = missing(mantra_cells())
    m_k3 = missing(k3_cells())
    if args.verbose:
        for name, cells, miss in [("mantra_arrays", mantra_cells(), m_mantra),
                                  ("hopse_k3_array", k3_cells(), m_k3)]:
            print(f"{name}: {len(cells) - len(miss)}/{len(cells)} complete")
            for i, path, rows in cells:
                if i in miss:
                    print(f"  [{i:2d}] MISSING {path.relative_to(REPO)}")
        return
    print(",".join(str(i) for i in m_mantra))
    print(",".join(str(i) for i in m_k3))


if __name__ == "__main__":
    main()
