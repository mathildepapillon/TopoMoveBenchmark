"""E3: participation-game GEA on the GraphXAI molecule datasets.

One-framework recompute of the Table-1 TopoSHAP row: explain the
locally-trained anchored-recipe GCCN under the PARTICIPATION game and
score GEA against the GraphXAI ground-truth motifs, using the validated
exp #1 protocol ported into the library:

  - lifting: CellCycleLiftingGT (cell-to-graph maps; toposhap.lifting_gt)
  - selection: ``smallest_gt_positive`` — the 50 smallest ground-truth-
    positive test complexes, sorted by (n_players, orig_index)
  - value: label-class logit (leg-1 convention), pool-faithful
  - GEA: max over GraphXAI candidate explanations of the Jaccard at
    ground-truth size (toposhap.metrics.gea)

``--game signal`` reproduces the frozen signal-game semantics
(CellMaskingGame, wiring intact) for harness validation against the
frozen leg-1 numbers before the semantics flips.

Usage:
    python molecule_participation_gea.py DATASET SEED [PASSES] [GAME]
Writes results/molecule_gea/{DATASET}_{GAME}_p{PASSES}_s{SEED}.json.
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
os.environ.setdefault(
    "TOPOSHAP_GRAPHXAI_DATA",
    str(REPO / "external" / "TopoBench" / "datasets" / "graphxai"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_attribution_figure import single_batch  # noqa: E402
from mantra_balanced_eval import R, find_ckpt, load_ckpt  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.explain import CellMaskingGame  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_logit,
)
from toposhap.metrics.faithfulness import random_control_gea  # noqa: E402
from toposhap.metrics.gea import (  # noqa: E402
    candidate_cell_masks,
    graph_explanation_accuracy,
    node_projected_gea,
)
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402
from toposhap.vocabulary import mask_to_coalition  # noqa: E402

N_EXPLAIN = 50  # leg-1 protocol: complexes_per_seed


def main() -> None:
    ds = sys.argv[1]
    seed = int(sys.argv[2])
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    game_kind = sys.argv[4] if len(sys.argv) > 4 else "participation"
    karm = sys.argv[5] if len(sys.argv) > 5 else "k3"
    assert game_kind in ("participation", "signal")
    assert karm in ("k3", "k6", "k11", "k11s")

    if karm == "k3":
        row = json.loads(
            (R / "anchored3_exact" / f"anchored3_{ds}_seed{seed}.jsonl")
            .read_text())
        ckpt_dir = (R / "anchored3_exact" / "runs" / f"{ds}_seed{seed}"
                    / "continue" / "checkpoints")
        sel_epoch = row.get("selected_epoch_in_continuation")
    else:
        base = R / f"molecule_top{karm}"
        rows = [json.loads(x) for x in
                (base / f"top{karm}_{ds}.jsonl")
                .read_text().splitlines() if x]
        row = next(r for r in rows if r["seed"] == seed)
        ckpt_dir = base / "runs" / f"{ds}_s{seed}" / "checkpoints"
        sel_epoch = row.get("selected_epoch")
    mask = row["mask"]

    from toposhap.vocabulary import NEIGHBORHOODS

    def compose(neigh, tag):
        return drv.compose_config(
            dataset=ds, seed=seed, neighborhoods=neigh,
            output_dir=str(R / "molecule_gea" / f"cfg_{ds}_s{seed}_{tag}"),
            model=(f"{drv.model_domain(ds)}/"
                   f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
            extra_overrides=[
                "transforms.graph2cell_lifting.transform_name="
                "CellCycleLiftingGT",
            ] + (["transforms.graph2cell_lifting.feature_lifting="
                  "StructuralOnes"] if karm == "k11s" else []),
        )

    # The anchored recipe composes with the FULL vocabulary, prunes the
    # model IN PLACE (prune_backbone_ renumbers kept routes
    # contiguously), then trains the continuation — so the checkpoint
    # only strict-loads into a full-composed model pruned BEFORE loading.
    cfg = compose(list(NEIGHBORHOODS), "full")
    pipe = drv.build_pipeline(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt(ckpt_dir, sel_epoch)
    state = torch.load(str(ckpt), map_location="cpu",
                       weights_only=False)["state_dict"]
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, mask)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # sanity: the GT lifting must not change what the model sees
    loader = pipe.datamodule.test_dataloader()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            b["model_state"] = "test"
            o = model.forward(b)
            o = model.process_outputs(model_out=o, batch=b)
            correct += int((o["logits"].argmax(-1) == o["labels"]).sum())
            total += int(o["labels"].numel())
    acc = correct / total
    print(f"{ds} s{seed}: test acc {acc:.4f} "
          f"(anchored row: {row.get('test_accuracy'):.4f})", flush=True)
    if abs(acc - row.get("test_accuracy", acc)) > 0.02:
        raise SystemExit("GT-lifted pipeline does not reproduce the "
                         "anchored checkpoint's test accuracy — refusing")

    source, _ = hydra.utils.instantiate(cfg.dataset.loader).load()

    # smallest_gt_positive selection over the test split
    test = pipe.datamodule.dataset_test
    entries = []
    for pos in range(len(test.data_lst)):
        d = test.data_lst[pos]
        if not bool(np.asarray(d.gt_positive).reshape(-1)[0]):
            continue
        n_players = sum(int(getattr(d, f"x_{r}").shape[0])
                        for r in (0, 1, 2) if hasattr(d, f"x_{r}"))
        orig = int(np.asarray(d.orig_index).reshape(-1)[0])
        entries.append((n_players, orig, pos))
    entries.sort()
    n_explain = int(os.environ.get("TOPOSHAP_GEA_LIMIT", str(N_EXPLAIN)))
    if n_explain <= 0:  # 0 = the FULL gt-positive test population
        n_explain = len(entries)
    selected = entries[:n_explain]
    selection_name = ("full_gt_positive" if len(selected) == len(entries)
                      else "smallest_gt_positive")
    # TOPOSHAP_GEA_SHARD="i/k" explains the strided slice selected[i::k]
    # (size-sorted, so strides balance cost); shard files carry a suffix
    # and the analysis concatenates them back into the full population.
    shard = os.environ.get("TOPOSHAP_GEA_SHARD", "")
    if shard:
        si, sk = (int(v) for v in shard.split("/"))
        selected = selected[si::sk]
    print(f"{len(entries)} gt-positive test complexes; explaining "
          f"{len(selected)}{f' (shard {shard})' if shard else ''} "
          f"(players {selected[0][0]}-{selected[-1][0]})",
          flush=True)

    out = {"dataset": ds, "seed": seed, "passes": passes, "game": game_kind,
           "k_arm": karm, "mask": mask, "ckpt": ckpt.name,
           "test_accuracy": acc, "shard": shard,
           "selection": selection_name, "n_population": len(entries),
           "records": []}
    out_dir = R / "molecule_gea"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if karm == "k3" else f"_{karm}"
    if selection_name == "full_gt_positive":
        suffix += "_full"
    if shard:
        suffix += f"_sh{shard.replace('/', 'of')}"
    out_file = out_dir / f"{ds}_{game_kind}_p{passes}_s{seed}{suffix}.json"

    for done, (n_players, orig, pos) in enumerate(selected):
        data = test.data_lst[pos]
        batch = single_batch(loader, pos).to(device)
        batch["model_state"] = "test"
        label = int(np.asarray(data.y).reshape(-1)[0])
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   if hasattr(batch, f"x_{r}")
                   for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
        n = len(players)
        n_cells = {r: int(getattr(batch, f"x_{r}").shape[0])
                   for r in (0, 1, 2) if hasattr(batch, f"x_{r}")}
        offset, acc_off = {}, 0
        for r in sorted(n_cells):
            offset[r] = acc_off
            acc_off += n_cells[r]

        model_fn = pool_faithful_logit(model, label)
        if game_kind == "participation":
            raw = CellParticipationGame(model_fn, batch, players,
                                        encoder=model.feature_encoder)
        else:
            raw = CellMaskingGame(model_fn, batch, players,
                                  encoder=model.feature_encoder)
        game = CachedGame(n_players=n, evaluate=raw)
        att = sampled_shapley(game, n, passes=passes, seed=0)
        phi = np.asarray(att.phi)

        candidates = source.gt_candidates[orig]
        gt_masks = candidate_cell_masks(data, candidates, offset, n_cells)
        best = graph_explanation_accuracy(phi, gt_masks)
        best_node = node_projected_gea(phi, data, candidates, offset,
                                       n_cells, gt_masks)
        rc_mean, rc_sd = random_control_gea(
            list(range(n)), set(np.where(gt_masks[best["candidate"]])[0]),
            best["gt_size"], seed=orig)

        out["records"].append({
            "orig_index": orig, "test_pos": pos, "label": label,
            "n_players": n, "gea": best["gea"],
            "gea_node": best_node["gea_node"],
            "gt_size": best["gt_size"], "gt_candidates": int(
                gt_masks.shape[0]),
            "random_gea_mean": rc_mean, "random_gea_sd": rc_sd,
            "v_full": float(game((1 << n) - 1)),
            "v_empty": float(game(0)),
            "evaluations": game.calls,
        })
        if (done + 1) % 10 == 0:
            print(f"  {done + 1}/{len(selected)}", flush=True)
            out_file.write_text(json.dumps(out))

    geas = [r["gea"] for r in out["records"]]
    gnode = [r["gea_node"] for r in out["records"]]
    out["gea_mean"] = float(np.mean(geas))
    out["gea_sd_graphs"] = float(np.std(geas))
    out["gea_node_mean"] = float(np.mean(gnode))
    out_file.write_text(json.dumps(out))
    print(f"GEA mean {out['gea_mean']:.4f} over {len(geas)} graphs; "
          f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
