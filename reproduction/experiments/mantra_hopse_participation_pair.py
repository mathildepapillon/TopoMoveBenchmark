"""HOPSE participation-game attributions on the matched MANTRA pair.

Encodings are functions of the complex, so the participation game
re-runs the HOPSE preprocessing transform on the detached complex at
every evaluation (HopseParticipationGame). Model: seed-43 sweep
val-best. Post-freeze, talk-only. Writes
results/mantra_attr_fig/hopse_participation_pair.json.
"""

from __future__ import annotations

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
from mantra_balanced_eval import DS, R, SCRATCH, find_ckpt, load_ckpt  # noqa: E402
from run_hopse_sweep import compose as sweep_compose  # noqa: E402
from run_hopse_sweep import sweep_configs  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    HopseParticipationGame,
    hopse_pool_faithful_margin,
)
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402

SEED = 43
# Sharding for wall-clock parallelism: run K jobs with PASSES/K passes
# under distinct SHARD_SEEDs and average phi (each shard is an unbiased
# estimator; see experiments/merge_hopse_shards.py).
PASSES = int(os.environ.get("TOPOSHAP_HOPSE_PASSES", "256"))
SHARD_SEED = int(os.environ.get("TOPOSHAP_HOPSE_SEED", "0"))
TARGET = os.environ.get("TOPOSHAP_HOPSE_TARGET", "")  # torus|klein|both
OUT_SUFFIX = os.environ.get("TOPOSHAP_HOPSE_OUT", "")


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def main() -> None:
    device = torch.device("cpu")  # transform re-runs are CPU-bound anyway
    # seed-43 val-best sweep config (as wired)
    best = None
    for spec in sweep_configs():
        p = R / "hopse_sweep_mantra" / f"sweep_{DS}_{spec['tag']}.jsonl"
        if not p.exists():
            continue
        for row in read_jsonl(p):
            if int(row["seed"]) != SEED:
                continue
            if best is None or row["val_accuracy"] > best[0]["val_accuracy"]:
                best = (row, spec)
    row, spec = best
    print("model:", spec["tag"], flush=True)
    cfg = sweep_compose(DS, SEED, spec["members"],
                        SCRATCH / f"hpart_{spec['tag']}")
    pipe = drv.build_pipeline(cfg)
    ckpt = find_ckpt(
        R / "hopse_sweep_mantra" / "runs" / f"{DS}_{spec['tag']}_seed{SEED}"
        / "checkpoints", row.get("selected_epoch"))
    load_ckpt(pipe.model, ckpt)
    model = pipe.model.to(device).eval()
    from omegaconf import OmegaConf
    from topobench.transforms.data_transform import DataTransform
    # A FRESH OmegaConf tree satisfies both encoder constraints at once:
    # lists stay ListConfig (HKdiagSE branches on `type(kernel_param) is
    # ListConfig` to compute pe_dim — a plain list corrupts its
    # empty-graph path) and struct mode is off (CombinedFE `.pop()`s its
    # params, which hydra's struct-locked nodes forbid). The game's
    # full-coalition guard verifies this construction reproduces the
    # stored encodings.
    tcfg = dict(OmegaConf.create(
        OmegaConf.to_container(cfg.transforms.hopse_encoding, resolve=True)))
    tcfg["device"] = "cpu"
    transform = DataTransform(**tcfg).transform

    model_fn = hopse_pool_faithful_margin(model, 0, 1)

    loader = pipe.datamodule.test_dataloader()
    out = {"seed": SEED, "tag": spec["tag"], "ckpt": ckpt.name,
           "passes": PASSES, "game": "participation (encodings recomputed)",
           "note": "margin = logit[non-orientable] - logit[orientable]"}
    from topobench.dataloader.dataload_dataset import DataloadDataset

    def coll(data_list):
        return loader.collate_fn([DataloadDataset(data_list)[0]])

    pair = [("torus", 243), ("klein", 73)]
    if TARGET in ("torus", "klein"):
        pair = [p for p in pair if p[0] == TARGET]
    out["shard_seed"] = SHARD_SEED
    for name, idx in pair:
        sample = loader.dataset.data_lst[idx]
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   for i in range(int(getattr(sample, f"x_{r}").shape[0]))]
        print(f"constructing game / {name} ...", flush=True)
        raw = HopseParticipationGame(
            model_fn, sample, coll, players, transform,
            batch_extra={"model_state": "test"})

        def counted(mask, _raw=raw, _c=[0]):
            _c[0] += 1
            if _c[0] % 1000 == 0:
                print(f"  {_c[0]} evals ...", flush=True)
            return _raw(mask)

        game = CachedGame(n_players=len(players), evaluate=counted)
        att = sampled_shapley(game, len(players), passes=PASSES,
                              seed=SHARD_SEED)
        full = (1 << len(players)) - 1
        out[name] = {"phi": [float(v) for v in att.phi],
                     "player_ranks": [p.rank for p in players],
                     "v_full": raw(full), "v_empty": raw(0),
                     "evaluations": game.calls, "test_index": idx}
        print(f"  done ({game.calls} evals)", flush=True)
    path = (R / "mantra_attr_fig"
            / f"hopse_participation_pair{OUT_SUFFIX}.json")
    path.write_text(json.dumps(out))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
