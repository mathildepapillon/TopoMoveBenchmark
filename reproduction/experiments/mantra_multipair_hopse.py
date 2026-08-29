"""E1+E2 (HOPSE arm): participation attributions + baselines on the
size-matched MANTRA sample, per-seed sweep pick, CPU-only.

Per complex: sampled Shapley under the HOPSE participation game
(encodings recomputed from the detached complex each evaluation),
random + occlusion baselines over the same players, and deletion
curves for all three rankings. gradient x input is out of scope for
HOPSE (the encodings are recomputed inputs, not differentiable leaves)
and is disclosed as such.

Usage: python mantra_multipair_hopse.py SEED
Writes results/mantra_attr_fig/multipair/hopse_s{SEED}.json.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import DS, R, SCRATCH, find_ckpt, load_ckpt  # noqa: E402
from run_hopse_sweep import compose as sweep_compose  # noqa: E402
from run_hopse_sweep import sweep_configs  # noqa: E402
from toposhap.baselines import occlusion_scores, random_scores  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    HopseParticipationGame,
    hopse_pool_faithful_margin,
)
from toposhap.metrics.deletion import deletion_curve  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402

PASSES = 128


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def main() -> None:
    seed = int(sys.argv[1])
    best = None
    for spec in sweep_configs():
        p = R / "hopse_sweep_mantra" / f"sweep_{DS}_{spec['tag']}.jsonl"
        if not p.exists():
            continue
        for row in read_jsonl(p):
            if row["seed"] == seed and (
                    best is None
                    or row["val_accuracy"] > best[0]["val_accuracy"]):
                best = (row, spec)
    row, spec = best
    print(f"seed {seed}: model {spec['tag']}", flush=True)

    cfg = sweep_compose(DS, seed, spec["members"],
                        SCRATCH / f"hmp_{spec['tag']}_s{seed}")
    pipe = drv.build_pipeline(cfg)
    ckpt = find_ckpt(
        R / "hopse_sweep_mantra" / "runs" / f"{DS}_{spec['tag']}_seed{seed}"
        / "checkpoints", row.get("selected_epoch"))
    load_ckpt(pipe.model, ckpt)
    model = pipe.model.to("cpu").eval()

    from omegaconf import OmegaConf
    from topobench.dataloader.dataload_dataset import DataloadDataset
    from topobench.transforms.data_transform import DataTransform
    tcfg = dict(OmegaConf.create(
        OmegaConf.to_container(cfg.transforms.hopse_encoding, resolve=True)))
    tcfg["device"] = "cpu"
    transform = DataTransform(**tcfg).transform
    loader = pipe.datamodule.test_dataloader()

    def coll(data_list):
        return loader.collate_fn([DataloadDataset(data_list)[0]])

    model_fn = hopse_pool_faithful_margin(model, 0, 1)

    pairs = json.loads(
        (R / "mantra_attr_fig" / "matched_pairs.json").read_text())
    sample = [(idx, int(y), p["counts"])
              for p in pairs["sample"]
              for y, idxs in p["sampled_indices"].items()
              for idx in idxs]
    limit = int(os.environ.get("TOPOSHAP_MULTIPAIR_LIMIT", "0"))
    if limit:
        sample = sample[:limit]
    print(f"{len(sample)} complexes", flush=True)

    out = {"seed": seed, "tag": spec["tag"], "passes": PASSES,
           "game": "participation (encodings recomputed, pool-faithful)",
           "note": "margin = logit[non-orientable] - logit[orientable]; "
                   "gradient x input not defined for recomputed encodings",
           "records": []}
    out_path = R / "mantra_attr_fig" / "multipair"
    out_path.mkdir(parents=True, exist_ok=True)

    for done, (idx, y, counts) in enumerate(sample):
        sample_data = loader.dataset.data_lst[idx]
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   for i in range(int(getattr(sample_data,
                                              f"x_{r}").shape[0]))]
        n = len(players)
        raw = HopseParticipationGame(
            model_fn, sample_data, coll, players, transform,
            batch_extra={"model_state": "test"})
        game = CachedGame(n_players=n, evaluate=raw)
        att = sampled_shapley(game, n, passes=PASSES, seed=0)
        phi = np.asarray(att.phi)

        occ = occlusion_scores(game, n)
        rnd = random_scores(n, seed=idx)

        v_full = game((1 << n) - 1)
        sgn = 1.0 if v_full >= 0 else -1.0
        curves = {}
        for name, scores in (("toposhap", phi), ("occlusion", occ),
                             ("random", rnd)):
            ranking = np.argsort(sgn * np.asarray(scores))[::-1]
            curves[name] = deletion_curve(game, ranking, n)

        out["records"].append({
            "test_index": idx, "label": y, "counts": counts,
            "model": "hopse",
            "phi": [float(v) for v in phi],
            "player_ranks": [p.rank for p in players],
            "v_full": float(v_full), "v_empty": float(game(0)),
            "occlusion": [float(v) for v in occ],
            "deletion_curves": curves,
            "evaluations": game.calls,
        })
        print(f"  {done + 1}/{len(sample)} (idx {idx}, "
              f"{game.calls} evals)", flush=True)
        (out_path / f"hopse_s{seed}.json").write_text(json.dumps(out))

    print("wrote", out_path / f"hopse_s{seed}.json", flush=True)


if __name__ == "__main__":
    main()
