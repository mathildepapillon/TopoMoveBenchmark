"""Loaders for the frozen result files (read these, do not recompute).

Every loader validates the schema probes documented in the project handoff, so
an ingested file that doesn't match what the analysis code expects fails
loudly at load time, not silently at figure time. All paths resolve relative
to ``data/frozen/`` unless absolute.

Expected inventory (see data/INGEST.md for provenance):

* analysis12.json         — per-seed picks/masks/tests
                            (per_dataset[ds].method_A.cells.guided)
* cost_wallclock12.json   — component seconds per dataset
* menu12.json             — menu per-seed val/test
                            (per_dataset[ds].sets[mask].per_seed)
* cheap_sim12.json        — pick-stability resampling
                            (per_dataset[ds].per_seed[seed].permutation[budget])
* menu_extraction12.json, smokegate12.json
* pooled.json             — 6-space landscape values
                            (retraining_pooled[ds]["values_mean"], 64 masks)
* cold_compare11.json     — warm-vs-cold continuation
* marker_arm14/           — #14 marker-arm checkpoint (plain + anchored,
                            5 seeds, per-seed masks, variance decomposition,
                            inline FLOPs), parquet/JSON
* cost_table13.parquet    — #13 Stage D unified cost table
"""

from __future__ import annotations

import json
from pathlib import Path

FROZEN_DIR = Path(__file__).resolve().parents[3] / "data" / "frozen"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else FROZEN_DIR / p


def _load_json(path: str | Path) -> dict:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not ingested yet — run scripts/ingest_results.py "
            "(see data/INGEST.md)"
        )
    with open(p) as f:
        return json.load(f)


def _require(obj: dict, dotted: str, filename: str) -> None:
    node = obj
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"{filename}: expected schema path '{dotted}' missing at "
                f"'{key}' — file does not match the handoff schema"
            )
        node = node[key]


def load_analysis12(path: str | Path = "analysis12.json") -> dict:
    """Per-seed guided picks/masks/tests from experiment #12."""
    data = _load_json(path)
    _require(data, "per_dataset", "analysis12.json")
    ds = next(iter(data["per_dataset"]))
    _require(
        data["per_dataset"][ds], "method_A.cells", f"analysis12.json[{ds}]"
    )
    return data


def load_cost_wallclock12(path: str | Path = "cost_wallclock12.json") -> dict:
    """Component wall-clock seconds (stem/game/continuation) per dataset."""
    return _load_json(path)


def load_menu12(path: str | Path = "menu12.json") -> dict:
    """TopoBench menu per-seed val/test per mask."""
    data = _load_json(path)
    _require(data, "per_dataset", "menu12.json")
    ds = next(iter(data["per_dataset"]))
    _require(data["per_dataset"][ds], "sets", f"menu12.json[{ds}]")
    return data


def load_cheap_sim12(path: str | Path = "cheap_sim12.json") -> dict:
    """Pick-stability resampling (per_seed[seed].permutation[budget])."""
    data = _load_json(path)
    _require(data, "per_dataset", "cheap_sim12.json")
    return data


def load_pooled9(path: str | Path = "pooled.json") -> dict:
    """6-space landscape values from experiment #9 (64 masks per dataset)."""
    data = _load_json(path)
    _require(data, "retraining_pooled", "pooled.json")
    return data


def load_cold_compare11(path: str | Path = "cold_compare11.json") -> dict:
    """Warm-vs-cold continuation comparison from experiment #11."""
    return _load_json(path)


def load_landscape_as_game(pooled: dict, dataset: str):
    """TabulatedGame over a dataset's landscape values_mean table.

    In the frozen #9 files ``values_mean`` is a positional list: index = mask
    over the ORIGINAL 6-neighborhood space (64 entries). CAUTION: the 6-space
    bit order is not recorded in the file and predates the widened
    11-neighborhood vocabulary — 6-space masks are NOT comparable to 11-space
    masks without the #9 player list. Size-based queries (argmax_at_size,
    popcounts) are order-free and safe.
    """
    from toposhap.shapley.games import TabulatedGame

    table = pooled["retraining_pooled"][dataset]["values_mean"]
    if isinstance(table, list):
        values = {mask: float(v) for mask, v in enumerate(table)}
    else:
        values = {int(mask): float(v) for mask, v in table.items()}
    n_players = max(values).bit_length()
    return TabulatedGame(n_players=n_players, values=values)


def load_marker_arm14(path: str | Path = "marker_arm14"):
    """#14 marker-arm checkpoint as a tidy frame (one row per dataset x
    selector x k x seed).

    Parses ``results/markers14.json`` (per_k_cells: {selector: {k: {seed:
    {val,test,mask}}}}). Selector names are normalised: the checkpoint calls
    the plain arm 'plain' in per_k_cells and 'guided' in component keys.

    Wall-clock per row = stem + sampled game + continuation, measured where
    recorded, with documented fallbacks:
    * seeds 42-44 reuse #12 stems/games -> cost_wallclock12 per_seed
      (stem5_seconds + 0.1 x exact game_seconds, the #12 sampled pricing);
      seeds 45-46 use #14's measured component_seconds.
    * anchored continuations not individually clocked fall back to the same
      seed's plain continuation at equal k (same epochs; identical when the
      picks agree, which is the common case).
    FLOPs are NaN pending the cost13 join (see docs/method-improvements.md).
    """
    import numpy as np
    import pandas as pd

    p = _resolve(path)
    markers = p / "results" / "markers14.json"
    if not markers.exists():
        raise FileNotFoundError(
            f"{markers} not ingested yet — the #14 marker-arm checkpoint is "
            "the gating input for the lead figure (see data/INGEST.md)"
        )
    m = json.loads(markers.read_text())
    try:
        cost12 = load_cost_wallclock12()["per_dataset"]
    except FileNotFoundError:
        cost12 = {}

    rows = []
    for ds, node in m["per_dataset"].items():
        comp = node.get("component_seconds", {})
        c12 = cost12.get(ds, {}).get("per_seed", {})

        def base_seconds(seed: str) -> float:
            if seed in comp.get("stems", {}):
                stem = comp["stems"][seed]
                game = comp["sampled_games"][seed]["seconds"]
                return stem + game
            if seed in c12:
                return (
                    c12[seed]["stem5_seconds"]
                    + 0.1 * c12[seed]["game_seconds"]
                )
            return float("nan")

        def cont_seconds(seed: str, selector: str, k: str) -> float:
            conts = comp.get("continuations", {})
            name = "guided" if selector == "plain" else selector
            direct = conts.get(f"s{seed}_{name}_k{k}")
            if direct is not None:
                return direct
            plain = conts.get(f"s{seed}_guided_k{k}")
            if plain is not None:
                return plain
            byk = c12.get(seed, {}).get("continued_seconds_by_k", {})
            return byk.get(k, float("nan"))

        for selector, by_k in node["per_k_cells"].items():
            for k, by_seed in by_k.items():
                for seed, leaf in by_seed.items():
                    rows.append(
                        dict(
                            dataset=ds,
                            seed=int(seed),
                            selector=selector,
                            k=int(k),
                            mask=int(leaf["mask"]),
                            val_accuracy=float(leaf["val"]),
                            test_accuracy=float(leaf["test"]),
                            wall_seconds=base_seconds(seed)
                            + cont_seconds(seed, selector, k),
                            flops=np.nan,
                        )
                    )
    return pd.DataFrame(rows)
