"""E1+E2: participation attributions + baselines on the size-matched
MANTRA sample, GCCN family, one seed per invocation.

For every complex in the deterministic 58-complex sample
(results/mantra_attr_fig/matched_pairs.json) and each of the three
GCCN-family models (menu CWN mask 26, ladder pick, anchored pick):

  - sampled Shapley under the participation game (PASSES passes)
  - baselines over the same players: random, occlusion (through the
    game cache), gradient x input on raw features
  - deletion curves under participation removal for all four rankings
    (occlusion + deletion evals mostly hit the Shapley run's cache)

Usage: python mantra_multipair_participation.py SEED
Writes results/mantra_attr_fig/multipair/gccn_family_s{SEED}.json.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import os

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_attribution_figure import single_batch  # noqa: E402
from mantra_balanced_eval import (  # noqa: E402
    DS,
    R,
    gccn_compose,
    gccn_ladder_paths,
    menu_paths,
)
from mantra_participation_pair import build, read_jsonl  # noqa: E402
from toposhap.baselines import (  # noqa: E402
    gradient_x_input,
    occlusion_scores,
    random_scores,
)
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_margin,
)
from toposhap.metrics.deletion import deletion_curve  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402

PASSES = 128
POS, NEG = 0, 1  # margin = logit[non-orientable] - logit[orientable]


def load_models(pipe, seed, device):
    models = {}
    menu_rows = [r for r in read_jsonl(menu_paths()[26]) if r["seed"] == seed]
    models["menu"] = build(
        pipe, 26, R / "gccn_menu_mantra" / "runs" / f"{DS}_mask26_s{seed}"
        / "checkpoints", "selected_epoch", menu_rows[0], device)[0]
    lrow = json.loads(gccn_ladder_paths()[seed].read_text())
    rung = lrow[lrow["selected"]]
    models["gccn"] = build(
        pipe, rung["mask"],
        R / "ladder_b2" / "runs" / f"{DS}_seed{seed}"
        / f"continue_{lrow['selected']}" / "checkpoints",
        "selected_epoch_in_continuation", rung, device)[0]
    arow = json.loads(
        (R / "anchored3_exact" / f"anchored3_{DS}_seed{seed}.jsonl")
        .read_text())
    models["anchored"] = build(
        pipe, arow["mask"],
        R / "anchored3_exact" / "runs" / f"{DS}_seed{seed}"
        / "continue" / "checkpoints",
        "selected_epoch_in_continuation", arow, device)[0]
    return models


def main() -> None:
    seed = int(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = json.loads(
        (R / "mantra_attr_fig" / "matched_pairs.json").read_text())
    sample = [(idx, int(y), p["counts"])
              for p in pairs["sample"]
              for y, idxs in p["sampled_indices"].items()
              for idx in idxs]
    limit = int(os.environ.get("TOPOSHAP_MULTIPAIR_LIMIT", "0"))
    if limit:
        sample = sample[:limit]
    print(f"seed {seed}: {len(sample)} complexes", flush=True)

    pipe = drv.build_pipeline(gccn_compose(seed, f"multipair{seed}"))
    loader = pipe.datamodule.test_dataloader()
    models = load_models(pipe, seed, device)

    out = {"seed": seed, "passes": PASSES,
           "game": "participation (pool-faithful)",
           "note": "margin = logit[non-orientable] - logit[orientable]",
           "records": []}
    out_path = R / "mantra_attr_fig" / "multipair"
    out_path.mkdir(parents=True, exist_ok=True)

    for done, (idx, y, counts) in enumerate(sample):
        batch = single_batch(loader, idx).to(device)
        batch["model_state"] = "test"
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
        n = len(players)
        for mname, model in models.items():
            # the game encodes ITS batch in place; gradient x input
            # encodes its own deepcopy — both need the raw batch
            gxi = gradient_x_input(model, batch, players, POS, NEG)
            b = copy.deepcopy(batch)
            model_fn = pool_faithful_margin(model, POS, NEG)
            raw = CellParticipationGame(model_fn, b, players,
                                        encoder=model.feature_encoder)
            game = CachedGame(n_players=n, evaluate=raw)
            att = sampled_shapley(game, n, passes=PASSES, seed=0)
            phi = np.asarray(att.phi)

            occ = occlusion_scores(game, n)
            rnd = random_scores(n, seed=idx)

            v_full = game((1 << n) - 1)
            sgn = 1.0 if v_full >= 0 else -1.0
            curves = {}
            for name, scores in (("toposhap", phi), ("occlusion", occ),
                                 ("gradxinput", gxi), ("random", rnd)):
                ranking = np.argsort(sgn * np.asarray(scores))[::-1]
                curves[name] = deletion_curve(game, ranking, n)

            out["records"].append({
                "test_index": idx, "label": y, "counts": counts,
                "model": mname,
                "phi": [float(v) for v in phi],
                "player_ranks": [p.rank for p in players],
                "v_full": float(v_full), "v_empty": float(game(0)),
                "occlusion": [float(v) for v in occ],
                "gradxinput": [float(v) for v in gxi],
                "deletion_curves": curves,
                "evaluations": game.calls,
            })
        if (done + 1) % 5 == 0:
            print(f"  {done + 1}/{len(sample)} complexes", flush=True)
            (out_path / f"gccn_family_s{seed}.json").write_text(
                json.dumps(out))

    (out_path / f"gccn_family_s{seed}.json").write_text(json.dumps(out))
    print("wrote", out_path / f"gccn_family_s{seed}.json", flush=True)


if __name__ == "__main__":
    main()
