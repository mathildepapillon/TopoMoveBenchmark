"""PROTOTYPE: coalition (selection) readout of the participation game.

Diagnosis (2026-08-27, fair-table analysis): per-player Shapley + top-k
is a precision-at-budget ranking readout — every motif cell must
independently out-rank every non-motif cell. SubgraphX wins because its
readout SEARCHES for one connected subgraph whose joint sufficiency is
maximal, which is exactly what GEA measures. The game is not the
problem; the readout is.

This prototypes the same-game fix: greedy sufficiency search over
connected sub-complexes. v(S) is the identical participation game value
(pool-faithful, label-class logit) used for Shapley attribution — one
game, two readouts. Connectivity comes from the complex's incidences
(a cell may join S if it shares a boundary/coboundary relation with S).

Runs on single-candidate ground-truth graphs (where the Shapley readout
is weakest) and prints coalition node-GEA vs the frozen Shapley
node-GEA from the completed k-arm run.

Usage: python prototype_coalition_readout.py DATASET SEED [N] [KARM]
Writes results/prototype_coalition/{ds}_{karm}_s{seed}.json.
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
from toposhap.metrics.gea import candidate_cell_masks, cells_to_nodes  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402


def player_adjacency(batch, offset, n_cells):
    """Adjacency between global player indices via boundary relations."""
    adj: dict[int, set[int]] = {}

    def link(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    inc1 = batch.incidence_1.coalesce()
    for v, e in inc1.indices().t().cpu().numpy():
        link(offset[0] + int(v), offset[1] + int(e))
    if 2 in n_cells and n_cells[2] > 0:
        inc2 = batch.incidence_2.coalesce()
        for e, f in inc2.indices().t().cpu().numpy():
            link(offset[1] + int(e), offset[2] + int(f))
    return adj


def greedy_coalition(game, n, budget, adj):
    """Grow a connected coalition greedily by sufficiency v(S + {c})."""
    S: set[int] = set()
    evals = 0
    while len(S) < budget:
        if S:
            frontier = set().union(*(adj.get(p, set()) for p in S)) - S
            if not frontier:
                frontier = set(range(n)) - S
        else:
            frontier = set(range(n))
        best_c, best_v = None, -np.inf
        for c in sorted(frontier):
            v = game(sum(1 << p for p in S | {c}))
            evals += 1
            if v > best_v:
                best_c, best_v = c, v
        S.add(best_c)
    return S, best_v, evals


def coalition_shapley(game, n, S, m, rng):
    """MC Shapley value of S fused into one player: mean over random
    contexts C of v(C | S) - v(C). SubgraphX's scoring, on our game."""
    rest = sorted(set(range(n)) - S)
    s_mask = sum(1 << p for p in S)
    tot = 0.0
    for _ in range(m):
        rng.shuffle(rest)
        k = int(rng.integers(0, len(rest) + 1))
        c_mask = sum(1 << p for p in rest[:k])
        tot += game(c_mask | s_mask) - game(c_mask)
    return tot / m


def beam_prune(game, n, budget, adj, width=4, mc=32, seed=0):
    """Top-down beam search on sufficiency, finalists re-ranked by the
    coalition-Shapley score. Adds SubgraphX's two search ingredients
    (non-greedy search, context-marginalized scoring) to greedy_prune."""
    rng = np.random.default_rng(seed)
    beams = [frozenset(range(n))]
    evals = 0
    size = n
    while size > budget:
        children = {}
        for S in beams:
            for c in S:
                child = S - {c}
                if child in children:
                    continue
                children[child] = game(sum(1 << p for p in child))
                evals += 1
        beams = sorted(children, key=children.get, reverse=True)[:width]
        size -= 1
    best_S, best_v = None, -np.inf
    for S in beams:
        v = coalition_shapley(game, n, set(S), mc, rng)
        evals += 2 * mc
        if v > best_v:
            best_S, best_v = set(S), v
    return best_S, best_v, evals


def greedy_prune(game, n, budget, adj):
    """Top-down: start from the full complex, greedily detach the cell
    whose removal least hurts v(S). Mirrors SubgraphX's from-the-top
    search: values stay in-distribution while the coalition is large,
    and what survives to the budget is the jointly sufficient core."""
    S = set(range(n))
    evals = 0
    v_S = game((1 << n) - 1)
    while len(S) > budget:
        best_c, best_v = None, -np.inf
        for c in sorted(S):
            v = game(sum(1 << p for p in S - {c}))
            evals += 1
            if v > best_v:
                best_c, best_v = c, v
        S.remove(best_c)
        v_S = best_v
    return S, v_S, evals


def main() -> None:
    ds = sys.argv[1]
    seed = int(sys.argv[2])
    n_graphs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    karm = sys.argv[4] if len(sys.argv) > 4 else "k6"
    mode = sys.argv[5] if len(sys.argv) > 5 else "grow"
    assert mode in ("grow", "prune", "beam", "shapley")
    value = sys.argv[6] if len(sys.argv) > 6 else "logit"
    assert value in ("logit", "margin")

    # subject checkpoint: same load path as molecule_participation_gea
    base = R / f"molecule_top{karm}"
    rows = [json.loads(x) for x in
            (base / f"top{karm}_{ds}.jsonl").read_text().splitlines() if x]
    row = next(r for r in rows if r["seed"] == seed)
    cfg = drv.compose_config(
        dataset=ds, seed=seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(R / "prototype_coalition" / f"cfg_{ds}_s{seed}"),
        model=(f"{drv.model_domain(ds)}/"
               f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
        extra_overrides=[
            "transforms.graph2cell_lifting.transform_name="
            "CellCycleLiftingGT",
        ],
    )
    pipe = drv.build_pipeline(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt(base / "runs" / f"{ds}_s{seed}" / "checkpoints",
                     row.get("selected_epoch"))
    state = torch.load(str(ckpt), map_location="cpu",
                       weights_only=False)["state_dict"]
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, row["mask"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    source, _ = hydra.utils.instantiate(cfg.dataset.loader).load()

    # frozen Shapley node-GEA for the same graphs (merged shards)
    tag = "_full" if karm == "k3" else f"_{karm}_full"
    shap = {}
    for f in (R / "molecule_gea").glob(
            f"{ds}_participation_p128_s{seed}{tag}*.json"):
        for r in json.loads(f.read_text())["records"]:
            shap[r["orig_index"]] = r["gea_node"]

    # single-candidate gt-positive test graphs, smallest first
    test = pipe.datamodule.dataset_test
    loader = pipe.datamodule.test_dataloader()
    entries = []
    for pos in range(len(test.data_lst)):
        d = test.data_lst[pos]
        if not bool(np.asarray(d.gt_positive).reshape(-1)[0]):
            continue
        orig = int(np.asarray(d.orig_index).reshape(-1)[0])
        if int(source.gt_candidates[orig]["node"].shape[0]) != 1:
            continue
        n_players = sum(int(getattr(d, f"x_{r}").shape[0])
                        for r in (0, 1, 2) if hasattr(d, f"x_{r}"))
        entries.append((n_players, orig, pos))
    entries.sort()
    entries = entries[:n_graphs]
    print(f"{len(entries)} single-candidate graphs", flush=True)

    out = {"dataset": ds, "seed": seed, "karm": karm, "mode": mode,
           "value": value, "records": []}
    for n_players, orig, pos in entries:
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

        if value == "margin":
            from toposhap.cells.participation import pool_faithful_margin

            # label-oriented margin: prediction-relevant score, invariant
            # to shifts that move both class logits together
            model_fn = pool_faithful_margin(model, pos_cls=label,
                                            neg_cls=1 - label)
        else:
            model_fn = pool_faithful_logit(model, label)
        raw = CellParticipationGame(
            model_fn, batch, players, encoder=model.feature_encoder)
        game = CachedGame(n_players=n, evaluate=raw)
        adj = player_adjacency(batch, offset, n_cells)

        gt_cells = candidate_cell_masks(data, source.gt_candidates[orig],
                                        offset, n_cells)[0]
        budget = int(gt_cells.sum())
        if mode == "shapley":
            # the fleet's attribution readout, under the chosen value fn
            from toposhap.shapley.sampled import sampled_shapley

            att = sampled_shapley(game, n, passes=128, seed=0)
            phi = np.asarray(att.phi)
            S = {int(i) for i in
                 np.argsort(-phi, kind="stable")[:budget]}
            v_S, evals = float("nan"), game.calls
        elif mode == "beam":
            S, v_S, evals = beam_prune(game, n, budget, adj)
        else:
            search = greedy_prune if mode == "prune" else greedy_coalition
            S, v_S, evals = search(game, n, budget, adj)

        # oracle check: does the game score the found coalition above the
        # ground-truth closure? If yes while Jaccard < 1, the subject —
        # not the search — diverges from the motif.
        gt_set = set(int(i) for i in np.where(gt_cells)[0])
        v_gt = game(sum(1 << p for p in gt_set))
        v_found = game(sum(1 << p for p in S))

        sel_nodes = cells_to_nodes(sorted(S), data, offset, n_cells)
        gt_nodes = set(np.where(
            source.gt_candidates[orig]["node"].cpu().numpy()[0] > 0)[0])
        inter = len(sel_nodes & gt_nodes)
        union = len(sel_nodes | gt_nodes)
        gea_node = inter / union if union else float("nan")
        gea_cell = (int((np.array(sorted(S)) != -1).sum() and
                        len(set(np.where(gt_cells)[0]) & S)) /
                    len(set(np.where(gt_cells)[0]) | S))
        rec = {"orig_index": orig, "n_players": n, "budget": budget,
               "gea_node_coalition": gea_node,
               "gea_cell_coalition": float(
                   len(set(np.where(gt_cells)[0]) & S) /
                   len(set(np.where(gt_cells)[0]) | S)),
               "gea_node_shapley": shap.get(orig),
               "v_coalition": float(v_S), "evals": evals,
               "v_found": float(v_found), "v_gt": float(v_gt)}
        out["records"].append(rec)
        print(f"  {orig}: coalition {gea_node:.3f} vs shapley "
              f"{shap.get(orig)} (budget {budget}, {evals} evals)",
              flush=True)

    rs = out["records"]
    co = [r["gea_node_coalition"] for r in rs]
    sh = [r["gea_node_shapley"] for r in rs if r["gea_node_shapley"]
          is not None]
    beats = [r for r in rs if r["v_found"] > r["v_gt"]]
    out["summary"] = {"coalition_mean": float(np.mean(co)),
                      "shapley_mean": float(np.mean(sh)) if sh else None,
                      "n": len(rs),
                      "found_beats_gt": len(beats),
                      "mean_v_found_minus_v_gt": float(np.mean(
                          [r["v_found"] - r["v_gt"] for r in rs]))}
    dest = R / "prototype_coalition"
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / f"{ds}_{karm}_s{seed}_{mode}_{value}.json"
    fp.write_text(json.dumps(out))
    print(f"coalition node-GEA {out['summary']['coalition_mean']:.3f} vs "
          f"shapley {out['summary']['shapley_mean']:.3f} over {len(rs)} "
          f"single-candidate graphs; wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
