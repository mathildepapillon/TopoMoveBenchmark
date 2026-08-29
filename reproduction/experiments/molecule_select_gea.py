"""TopoSHAP-Select GEA fleet: coalition readout at full population.

Identical protocol to experiments/molecule_participation_gea.py (same
subjects, same GT lifting, same accuracy gate, same populations, same
scoring convention: per-candidate budget = that candidate's cell count,
max over candidates), with the Shapley readout replaced by the Select
readout (toposhap.cells.select.greedy_prune_schedule) on the SAME
participation game. Also records the oracle values v(found) and
v(gt closure) per candidate for the model-motif alignment figure.

Usage:
    python molecule_select_gea.py DATASET SEED [KARM]
Env: TOPOSHAP_GEA_LIMIT (0/unset = full population),
     TOPOSHAP_GEA_SHARD="i/k" (strided shard, merged at analysis).
Writes results/molecule_select/{ds}_select_s{seed}_{karm}_full*.json.
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
from mantra_balanced_eval import R, find_ckpt  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_logit,
)
from toposhap.cells.select import greedy_prune_schedule  # noqa: E402
from toposhap.metrics.gea import candidate_cell_masks, cells_to_nodes  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402


def main() -> None:
    ds = sys.argv[1]
    seed = int(sys.argv[2])
    karm = sys.argv[3] if len(sys.argv) > 3 else "k6"
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

    cfg = drv.compose_config(
        dataset=ds, seed=seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(R / "molecule_select" / f"cfg_{ds}_s{seed}_{karm}"),
        model=(f"{drv.model_domain(ds)}/"
               f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
        extra_overrides=[
            "transforms.graph2cell_lifting.transform_name="
            "CellCycleLiftingGT",
        ] + (["transforms.graph2cell_lifting.feature_lifting="
              "StructuralOnes"] if karm == "k11s" else []),
    )
    pipe = drv.build_pipeline(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt(ckpt_dir, sel_epoch)
    state = torch.load(str(ckpt), map_location="cpu",
                       weights_only=False)["state_dict"]
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, row["mask"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

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
    print(f"{ds} {karm} s{seed}: test acc {acc:.4f} "
          f"(row: {row.get('test_accuracy'):.4f})", flush=True)
    if abs(acc - row.get("test_accuracy", acc)) > 0.02:
        raise SystemExit("checkpoint does not reproduce the trained "
                         "accuracy — refusing")

    source, _ = hydra.utils.instantiate(cfg.dataset.loader).load()

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
    limit = int(os.environ.get("TOPOSHAP_GEA_LIMIT", "0"))
    if limit > 0:
        entries = entries[:limit]
    shard = os.environ.get("TOPOSHAP_GEA_SHARD", "")
    if shard:
        si, sk = (int(v) for v in shard.split("/"))
        entries = entries[si::sk]
    print(f"explaining {len(entries)} complexes"
          f"{f' (shard {shard})' if shard else ''}", flush=True)

    out = {"dataset": ds, "seed": seed, "karm": karm, "method": "select",
           "mask": row["mask"], "test_accuracy": acc, "shard": shard,
           "records": []}
    out_dir = R / "molecule_select"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_full" if limit <= 0 else f"_l{limit}"
    if bool(int(os.environ.get("TOPOSHAP_LIFT_CONSISTENT", "0"))):
        suffix += "_lc"
        out["semantics"] = "lift_consistent"
    if shard:
        suffix += f"_sh{shard.replace('/', 'of')}"
    out_file = out_dir / f"{ds}_select_s{seed}_{karm}{suffix}.json"

    for done, (n_players, orig, pos) in enumerate(entries):
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

        lift_consistent = bool(int(
            os.environ.get("TOPOSHAP_LIFT_CONSISTENT", "0")))
        raw = CellParticipationGame(
            pool_faithful_logit(model, label), batch, players,
            encoder=model.feature_encoder,
            lift_consistent=lift_consistent)
        game = CachedGame(n_players=n, evaluate=raw)

        gt_masks = candidate_cell_masks(data, source.gt_candidates[orig],
                                        offset, n_cells)
        node_masks = (source.gt_candidates[orig]["node"].cpu().numpy()
                      > 0)
        budgets = [int(gt_masks[c].sum()) for c in range(gt_masks.shape[0])]
        sched = greedy_prune_schedule(game, n, budgets)

        def rank_of(p):
            return max(r for r in offset if offset[r] <= p)

        best_cell, best_node, best_c = float("nan"), float("nan"), -1
        best_comp = None
        oracle = []
        for c in range(gt_masks.shape[0]):
            b = budgets[c]
            if b <= 0 or b not in sched:
                continue
            S, v_S = sched[b]
            gt_set = set(int(i) for i in np.where(gt_masks[c])[0])
            jac_cell = len(gt_set & S) / len(gt_set | S)
            sel_nodes = cells_to_nodes(sorted(S), data, offset, n_cells)
            gt_nodes = set(int(i) for i in np.where(node_masks[c])[0])
            u = len(sel_nodes | gt_nodes)
            jac_node = len(sel_nodes & gt_nodes) / u if u else float("nan")
            v_gt = game(sum(1 << p for p in gt_set))
            oracle.append({"candidate": c, "v_found": float(v_S),
                           "v_gt": float(v_gt)})
            if np.isnan(best_node) or jac_node > best_node:
                best_node, best_cell, best_c = jac_node, jac_cell, c
                comp = {r: 0 for r in n_cells}
                inter = {r: 0 for r in n_cells}
                gtc = {r: 0 for r in n_cells}
                for p in S:
                    comp[rank_of(p)] += 1
                for p in gt_set:
                    gtc[rank_of(p)] += 1
                    if p in S:
                        inter[rank_of(p)] += 1
                best_comp = {"found_per_rank": comp,
                             "gt_per_rank": gtc,
                             "overlap_per_rank": inter}

        out["records"].append({
            "orig_index": orig, "test_pos": pos, "label": label,
            "n_players": n, "gea": best_cell, "gea_node": best_node,
            "candidate": best_c, "gt_candidates": int(gt_masks.shape[0]),
            "composition": best_comp,
            "oracle": oracle, "evaluations": game.calls,
        })
        if (done + 1) % 10 == 0:
            print(f"  {done + 1}/{len(entries)}", flush=True)
            out_file.write_text(json.dumps(out))

    gn = [r["gea_node"] for r in out["records"]]
    out["gea_node_mean"] = float(np.nanmean(gn))
    out_file.write_text(json.dumps(out))
    print(f"select node-GEA mean {out['gea_node_mean']:.4f} over "
          f"{len(gn)}; wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
