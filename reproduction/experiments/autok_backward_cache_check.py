#!/usr/bin/env python
"""Backward-elimination picks computed directly on the #14 cached games.

For the two seeds with cached games (45, 46), replay the backward-elimination
rule on data/frozen/marker_arm14/results/cheap/sampled_<dataset>_seed<s>_E5.npz
wherever the leave-one-out masks the path needs are present. Coverage is
partial by construction (the caches are permutation-sampled), so at each step
only members whose leave-one-out mask is cached are candidates; missing
candidates are recorded. Purely tabular — no GPU, no training.

    python experiments/autok_backward_cache_check.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from toposhap.vocabulary import FULL_MASK, N_PLAYERS  # noqa: E402

CHEAP = REPO / "data" / "frozen" / "marker_arm14" / "results" / "cheap"
DATASETS = [
    "benzene",
    "fluoride_carbonyl",
    "mutagenicity_gxai",
    "NCI1",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
]
#: Validation-set sizes measured by run 1 (m_val in results/autok JSONLs).
M_VAL = {
    "benzene": 1800,
    "fluoride_carbonyl": 1300,
    "mutagenicity_gxai": 265,
    "NCI1": 1027,
    "cocitation_cora": 677,
    "cocitation_citeseer": 831,
    "cocitation_pubmed": 4929,
}


def backward_on_table(table: dict[int, float], threshold: float):
    """Backward elimination on a partial game table.

    Returns (mask, trace, stopped_reason, missing_log): candidates whose
    leave-one-out mask is absent from the table are skipped and logged.
    """
    mask = FULL_MASK
    base = table[FULL_MASK]
    trace, missing_log = [], []
    while bin(mask).count("1") > 1:
        best_bit, best_value = None, -math.inf
        missing = []
        for i in range(N_PLAYERS):
            if not mask >> i & 1:
                continue
            cand = mask & ~(1 << i)
            if cand not in table:
                missing.append(i)
                continue
            if table[cand] > best_value:
                best_bit, best_value = i, table[cand]
        if missing:
            missing_log.append({"at_mask": mask, "missing_bits": missing})
        if best_bit is None:
            return mask, trace, "no_cached_candidates", missing_log
        if base - best_value > threshold:
            return mask, trace, "stopped_by_rule", missing_log
        mask &= ~(1 << best_bit)
        trace.append((best_bit, float(best_value)))
        base = table[mask]
    return mask, trace, "reached_k1", missing_log


def main() -> None:
    out = {}
    for dataset in DATASETS:
        for seed in (45, 46):
            path = CHEAP / f"sampled_{dataset}_seed{seed}_E5.npz"
            if not path.exists():
                print(f"{dataset} s{seed}: cache missing, skipped")
                continue
            blob = np.load(path, allow_pickle=True)
            table = {
                int(m): float(v)
                for m, v in zip(blob["eval_masks"], blob["eval_values"])
            }
            p = table[FULL_MASK]
            se = math.sqrt(p * (1 - p) / M_VAL[dataset])
            mask, trace, reason, missing = backward_on_table(table, se)
            n_loo_full = sum(
                1 for i in range(N_PLAYERS)
                if (FULL_MASK & ~(1 << i)) in table
            )
            out[f"{dataset}_seed{seed}"] = {
                "dataset": dataset,
                "seed": seed,
                "mask": mask,
                "k": bin(mask).count("1"),
                "v_full": p,
                "binomial_se": se,
                "trace": trace,
                "terminated": reason,
                "n_cached_masks": len(table),
                "n_leave_one_out_of_full_cached": n_loo_full,
                "missing_candidates_along_path": missing,
            }
            print(
                f"{dataset} s{seed}: mask={mask} k={bin(mask).count('1')} "
                f"({reason}); {n_loo_full}/11 full-coalition leave-one-outs "
                f"cached; {len(missing)} steps had uncached candidates"
            )
    dest = REPO / "results" / "autok_backward" / "cache_check14.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
