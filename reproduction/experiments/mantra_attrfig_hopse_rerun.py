"""HOPSE pair attributions, redone at the encoding interface.

Job 419349's HOPSE cell game was degenerate: ``CellMaskingGame`` masks
``x_{rank}``, which HOPSE's wrapper never reads (v_full == v_empty on
both pair complexes, all 54 phi exactly 0). This rerun explains the same
matched-count test pair with ``HopseCellMaskingGame`` — encode once,
zero the masked cell's row in every encoded per-hop tensor of its rank.
Menu/GCCN results in ``attr_fig_data.json`` are valid and untouched; the
figure builder merges this file in at build time.

Validation: the unmasked margin v(full) must reproduce the value recorded
by job 419349 (same checkpoint, same margin definition) — a hard assert.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import torch  # noqa: E402

from mantra_attribution_figure import (  # noqa: E402
    OUT,
    PASSES,
    RANKS,
    SEED,
    build_hopse,
    make_model_fn,
    single_batch,
)
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.explain import HopseCellMaskingGame  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402


def main() -> None:
    src = json.loads((OUT / "attr_fig_data.json").read_text())
    # Job 419349 ran with pos_cls = majority; its role names are swapped
    # relative to true semantics (see the frozen JSON's provenance notes).
    # We keep ITS margin convention so v(full) can be asserted against the
    # recorded values; the figure builder owns the semantic flip.
    pos_cls = int(src["orientable_class"])
    neg_cls = 1 - pos_cls

    device = torch.device("cpu")
    pipe, meta = build_hopse(device)
    model = pipe.model
    loader = pipe.datamodule.test_dataloader()
    max_hop = int(model.feature_encoder.hops)

    results = {
        "seed": SEED,
        "meta": meta,
        "max_hop": max_hop,
        "game": "HopseCellMaskingGame (encoded per-hop row masking, zeros)",
        "note": ("roles inherit attr_fig_data.json's SWAPPED naming: "
                 "'orientable' is truly the NON-orientable complex; the "
                 "margin is logit[majority] - logit[minority]"),
    }
    for role in ("orientable", "non_orientable"):
        entry = src["pair"][role]
        idx = int(entry["test_index"])
        batch = single_batch(loader, idx)
        players = [CellPlayer(rank=r, index=i)
                   for r in RANKS
                   for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
        fn = make_model_fn(model, pos_cls, neg_cls)
        raw = HopseCellMaskingGame(
            fn, batch, players, encoder=model.feature_encoder,
            max_hop=max_hop)
        game = CachedGame(n_players=len(players), evaluate=raw)
        full = (1 << len(players)) - 1
        v_full, v_empty = raw(full), raw(0)
        recorded = entry["hopse"]["v_full"]
        if abs(v_full - recorded) > 1e-3:
            raise AssertionError(
                f"{role}: rebuilt v_full {v_full:.6f} != recorded "
                f"{recorded:.6f} — wrong checkpoint or margin")
        if abs(v_full - v_empty) < 1e-6:
            raise AssertionError(f"{role}: game still degenerate")
        att = sampled_shapley(game, len(players), passes=PASSES, seed=0)
        results[role] = {
            "test_index": idx,
            "phi": [float(v) for v in att.phi],
            "player_ranks": [p.rank for p in players],
            "v_full": v_full,
            "v_empty": v_empty,
            "evaluations": game.calls,
            "passes": PASSES,
            "baseline": raw.baseline_name,
        }
        print(f"{role}: v_full={v_full:.4f} v_empty={v_empty:.4f} "
              f"sum_phi={sum(results[role]['phi']):.4f} "
              f"evals={game.calls}", flush=True)

    out_path = OUT / "hopse_pair_rerun.json"
    out_path.write_text(json.dumps(results))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
