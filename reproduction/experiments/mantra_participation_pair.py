"""Participation-game cell attributions on the matched MANTRA pair.

Recomputes the talk's attribution panels under detachment semantics
(:class:`toposhap.cells.participation.CellParticipationGame`) for the
three GCCN-family models — menu val-best (CWN, mask 26), ladder pick
(seed 43, mask 1024), anchored pick (mask 1029) — on test complexes
243 (torus) and 73 (Klein bottle). HOPSE is phase 2 (its participation
game must recompute encodings per coalition).

Post-freeze, presentation/method-development output:
results/mantra_attr_fig/participation_pair.json.
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

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_attribution_figure import single_batch  # noqa: E402
from mantra_balanced_eval import (  # noqa: E402
    DS,
    R,
    find_ckpt,
    gccn_compose,
    gccn_ladder_paths,
    load_ckpt,
    menu_paths,
)
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_margin,
)
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402

SEED = 43
PASSES = 512
RANKS = (0, 1, 2)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def build(pipe, mask, ckpt_dir, epoch_key, row, device):
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, mask)
    ckpt = find_ckpt(ckpt_dir, row.get(epoch_key))
    load_ckpt(model, ckpt)
    model.to(device).eval()
    return model, ckpt.name


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = drv.build_pipeline(gccn_compose(SEED, "partpair"))
    loader = pipe.datamodule.test_dataloader()

    models = {}
    menu_rows = [r for r in read_jsonl(menu_paths()[26]) if r["seed"] == SEED]
    models["menu"] = build(
        pipe, 26, R / "gccn_menu_mantra" / "runs" / f"{DS}_mask26_s{SEED}"
        / "checkpoints", "selected_epoch", menu_rows[0], device) + (26,)
    lrow = json.loads(gccn_ladder_paths()[SEED].read_text())
    rung = lrow[lrow["selected"]]
    models["gccn"] = build(
        pipe, rung["mask"],
        R / "ladder_b2" / "runs" / f"{DS}_seed{SEED}"
        / f"continue_{lrow['selected']}" / "checkpoints",
        "selected_epoch_in_continuation", rung, device) + (rung["mask"],)
    arow = json.loads(
        (R / "anchored3_exact"
         / f"anchored3_{DS}_seed{SEED}.jsonl").read_text())
    models["anchored"] = build(
        pipe, arow["mask"],
        R / "anchored3_exact" / "runs" / f"{DS}_seed{SEED}"
        / "continue" / "checkpoints",
        "selected_epoch_in_continuation", arow, device) + (arow["mask"],)

    pos_cls, neg_cls = 0, 1  # majority (non-orientable) minus minority
    out = {"seed": SEED, "passes": PASSES, "game": "participation/detachment",
           "note": "margin = logit[non-orientable] - logit[orientable]"}
    for name, (model, ckpt_name, mask) in models.items():
        model_fn = pool_faithful_margin(model, pos_cls, neg_cls)

        entry = {"mask": mask, "ckpt": ckpt_name}
        for cname, idx in (("torus", 243), ("klein", 73)):
            batch = single_batch(loader, idx).to(device)
            batch["model_state"] = "test"
            players = [CellPlayer(rank=r, index=i) for r in RANKS
                       for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
            print(f"{name} / {cname}: constructing game ...", flush=True)
            raw = CellParticipationGame(
                model_fn, batch, players, encoder=model.feature_encoder)
            game = CachedGame(n_players=len(players), evaluate=raw)
            att = sampled_shapley(game, len(players), passes=PASSES, seed=0)
            full = (1 << len(players)) - 1
            entry[cname] = {
                "phi": [float(v) for v in att.phi],
                "player_ranks": [p.rank for p in players],
                "v_full": raw(full),
                "v_empty": raw(0),
                "evaluations": game.calls,
                "test_index": idx,
            }
            print(f"  done ({game.calls} evals)", flush=True)
        out[name] = entry
    path = R / "mantra_attr_fig" / "participation_pair.json"
    path.write_text(json.dumps(out))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
