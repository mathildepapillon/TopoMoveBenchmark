"""The unified cost table: one row per run, across all approaches.

Parquet schema (Stage D, experiment #13). Every accuracy claim in the paper is
budget-matched through this table; every marker in the lead figure gets its
x-position from here (FLOPs axis; wall-clock retained for the appendix).

Approaches: 'recipe' (stem+game+continue, one row per seed x k x selector),
'menu' (per candidate set x seed), 'sweep' (per swept architecture x seed),
'exact_game' (exhaustive retraining game pricing), 'landscape' (background).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

COLUMNS = [
    "approach",
    "dataset",
    "seed",
    "selector",
    "k",
    "mask",
    "val_accuracy",
    "test_accuracy",
    "wall_seconds",
    "flops",
    "flops_torch",
    "flops_shim",
    "stem_seconds",
    "game_seconds",
    "continue_seconds",
    "game_evaluations",
    "epochs",
    "notes",
]


@dataclass
class CostRow:
    approach: str
    dataset: str
    seed: int
    selector: str | None = None
    k: int | None = None
    mask: int | None = None
    val_accuracy: float | None = None
    test_accuracy: float | None = None
    wall_seconds: float | None = None
    flops: int | None = None
    flops_torch: int | None = None
    flops_shim: int | None = None
    stem_seconds: float | None = None
    game_seconds: float | None = None
    continue_seconds: float | None = None
    game_evaluations: int | None = None
    epochs: int | None = None
    notes: str = ""


class CostLedger:
    """Append-only parquet-backed cost table."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: list[dict] = []
        if self.path.exists():
            self._rows = pd.read_parquet(self.path).to_dict("records")

    def append(self, row: CostRow) -> None:
        d = asdict(row)
        if row.flops is None and (
            row.flops_torch is not None or row.flops_shim is not None
        ):
            d["flops"] = (row.flops_torch or 0) + (row.flops_shim or 0)
        self._rows.append(d)

    def flush(self) -> Path:
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path, index=False)
        return self.path

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=COLUMNS)


def training_run_equivalents(
    df: pd.DataFrame, dataset: str, approach_seconds: float
) -> float:
    """Cost in units of one mean from-scratch training run on this dataset.

    The honest denominator: mean wall-clock of full from-scratch runs recorded
    in the table for this dataset (sweep/menu rows with no stem split).
    """
    full_runs = df[
        (df.dataset == dataset)
        & (df.approach.isin(["sweep", "menu"]))
        & df.wall_seconds.notna()
    ]
    if full_runs.empty:
        raise ValueError(f"no full-run cost baseline for {dataset}")
    return approach_seconds / float(full_runs.wall_seconds.mean())
