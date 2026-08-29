"""Balanced-accuracy patch for the MANTRA B=1 (autok backward) rows.

Same convention as experiments/mantra_balanced_eval.py (whose helpers
this reuses): reload the run's best checkpoint, compute balanced
metrics on val and test, patch the JSONL in place add-only, and record
the recomputed plain accuracy as a load-fidelity cross-check.
Selection used plain val accuracy as wired; balanced never re-selects.

    python scripts/patch_b1_mantra_balanced.py     # needs a GPU node
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from mantra_balanced_eval import (  # noqa: E402  (applies patches on import)
    NOTE,
    eval_gccn_pruned,
)
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

DS = "mantra_orientation"
SEEDS = [42, 43, 44]
R = REPO / "results" / "autok_backward"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        path = R / f"autok_backward_{DS}_seed{seed}.jsonl"
        row = json.loads(path.read_text().splitlines()[0])
        if "test_balanced_accuracy" in row:
            print(f"skip seed {seed}: already patched")
            continue
        cfg = drv.compose_config(
            dataset=DS, seed=seed, neighborhoods=list(NEIGHBORHOODS),
            output_dir=str(REPO / "results" / "balanced_eval_runs"
                           / f"autok_backward_{DS}_seed{seed}"),
            model=f"{drv.model_domain(DS)}/{row['model']}",
        )
        pipe = drv.build_pipeline(cfg)
        val, test = eval_gccn_pruned(
            pipe, int(row["mask"]), Path(row["best_model_path"]), device)
        row.update({
            "val_balanced_accuracy": float(val["balanced"]),
            "test_balanced_accuracy": float(test["balanced"]),
            "val_auroc": val["auroc"],
            "test_auroc": test["auroc"],
            "balanced_eval": {
                "note": NOTE,
                "val_plain_recomputed": float(val["plain"]),
                "test_plain_recomputed": float(test["plain"]),
                "val_plain_recorded": float(row["val_accuracy"]),
                "test_plain_recorded": float(row["test_accuracy"]),
            },
        })
        path.write_text(json.dumps(row) + "\n")
        print(f"seed {seed}: bal test {test['balanced']:.4f} "
              f"(plain recomputed {test['plain']:.4f} vs recorded "
              f"{row['test_accuracy']:.4f})")


if __name__ == "__main__":
    main()
