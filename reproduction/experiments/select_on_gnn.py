"""TopoSHAP-Select on the baseline GNNs: the special-case closure.

A graph is a rank-1 complex, so the participation game reduces to
induced-subgraph node removal, and the Select readout becomes
SubgraphX's search problem. Running OUR implementation on the SAME
GIN/GCN subjects the baseline suite explains tests the special-case
claim empirically: if Select-on-GIN matches SubgraphX-on-GIN, table
differences between TopoSHAP and SubgraphX are attributable to the
explained subject, not the explainer.

Protocol mirrors run_graph_baselines.py exactly: same checkpoints,
accuracy gate, full gt-positive population, per-candidate budget =
candidate node count (nested prune schedule), max over candidates.

Usage: python select_on_gnn.py DATASET ARCH SEED [METHOD] [SEMANTICS]
  METHOD    select (default) — output 2, greedy prune;
            shapley — output 1, 128-pass per-node Shapley + top-k
  SEMANTICS participation (default) — induced-subgraph hard removal;
            zerofill — features of absent nodes zeroed, edges kept
            (SubgraphX's convention; the semantics ablation)
Env: TOPOSHAP_GRAPH_BASELINE_LIMIT, TOPOSHAP_GRAPH_BASELINE_SHARD.
Writes results/graph_baselines/select_gnn_{ds}_{arch}_s{seed}*.json
(default combo) or gnn_{method}_{semantics}_{ds}_{arch}_s{seed}*.json.
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
os.environ.setdefault(
    "TOPOSHAP_GRAPHXAI_DATA",
    str(REPO / "external" / "TopoBench" / "datasets" / "graphxai"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_graph_baselines import GraphLogitAdapter, load_model  # noqa: E402
from toposhap.cells.select import greedy_prune_schedule  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402


def induced_subgraph_game(adapter, x, edge_index, label, device):
    """v(S) = label-class logit on the node-induced subgraph (absent
    nodes and their incident edges removed) — the participation game on
    a rank-1 complex."""
    n = int(x.shape[0])
    ei = edge_index

    def evaluate(mask: int) -> float:
        keep = torch.tensor([(mask >> i) & 1 for i in range(n)],
                            dtype=torch.bool, device=device)
        if not bool(keep.any()):
            xs = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=device)
            es = torch.empty((2, 0), dtype=ei.dtype, device=device)
            bv = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                return float(adapter(xs, es, batch=bv)[0, label])
        idx = torch.where(keep)[0]
        remap = torch.full((n,), -1, dtype=torch.long, device=device)
        remap[idx] = torch.arange(idx.numel(), device=device)
        e_keep = keep[ei[0]] & keep[ei[1]]
        es = remap[ei[:, e_keep]]
        xs = x[idx]
        bv = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        with torch.no_grad():
            return float(adapter(xs, es, batch=bv)[0, label])

    return evaluate


def zerofill_game(adapter, x, edge_index, label, device):
    """v(S) = label-class logit with absent nodes' features zeroed and
    the graph structure kept — SubgraphX's zero_filling convention."""
    n = int(x.shape[0])
    bv = torch.zeros(n, dtype=torch.long, device=device)

    def evaluate(mask: int) -> float:
        keep = torch.tensor([(mask >> i) & 1 for i in range(n)],
                            dtype=x.dtype, device=device)
        with torch.no_grad():
            return float(adapter(x * keep[:, None], edge_index,
                                 batch=bv)[0, label])

    return evaluate


def main() -> None:
    ds, arch, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    method = sys.argv[4] if len(sys.argv) > 4 else "select"
    semantics = sys.argv[5] if len(sys.argv) > 5 else "participation"
    assert method in ("select", "shapley") and semantics in (
        "participation", "zerofill")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, pipe, row, acc = load_model(ds, arch, seed, device)
    adapter = GraphLogitAdapter(pipe.model).to(device).eval()
    source, _ = hydra.utils.instantiate(cfg.dataset.loader).load()

    test = pipe.datamodule.dataset_test
    entries = []
    for pos in range(len(test.data_lst)):
        d = test.data_lst[pos]
        if bool(np.asarray(d.gt_positive).reshape(-1)[0]):
            entries.append((int(np.asarray(d.orig_index).reshape(-1)[0]),
                            pos))
    entries.sort()
    limit = int(os.environ.get("TOPOSHAP_GRAPH_BASELINE_LIMIT", "0"))
    if limit > 0:
        entries = entries[:limit]
    shard = os.environ.get("TOPOSHAP_GRAPH_BASELINE_SHARD", "")
    if shard:
        si, sk = (int(v) for v in shard.split("/"))
        entries = entries[si::sk]
    print(f"select-on-{arch}: {len(entries)} molecules"
          f"{f' (shard {shard})' if shard else ''}", flush=True)

    out = {"dataset": ds, "arch": arch, "seed": seed,
           "method": method, "semantics": semantics,
           "test_accuracy": acc, "shard": shard, "records": []}
    out_dir = REPO / "results" / "graph_baselines"
    suffix = f"_l{limit}" if limit > 0 else ""
    if shard:
        suffix += f"_sh{shard.replace('/', 'of')}"
    if method == "select" and semantics == "participation":
        stem = f"select_gnn_{ds}"
    else:
        stem = f"gnn_{method}_{semantics}_{ds}"
    out_file = out_dir / f"{stem}_{arch}_s{seed}{suffix}.json"

    for done, (orig, pos) in enumerate(entries):
        d = test.data_lst[pos]
        x = d.x.to(device)
        ei = d.edge_index.to(device)
        n = int(x.shape[0])
        label = int(np.asarray(d.y).reshape(-1)[0])
        gt_masks = source.gt_candidates[orig]["node"].cpu().numpy() > 0

        game_fn = (induced_subgraph_game if semantics == "participation"
                   else zerofill_game)
        game = CachedGame(
            n_players=n,
            evaluate=game_fn(adapter, x, ei, label, device))
        budgets = [int(m.sum()) for m in gt_masks]

        oracle = []
        if method == "shapley":
            from toposhap.metrics.gea import graph_explanation_accuracy
            from toposhap.shapley.sampled import sampled_shapley

            att = sampled_shapley(game, n, passes=128, seed=0)
            phi = np.asarray(att.phi)
            r = graph_explanation_accuracy(phi, gt_masks)
            best, best_c = r["gea"], r["candidate"]
        else:
            sched = greedy_prune_schedule(game, n, budgets)
            best, best_c = float("nan"), -1
            for c in range(gt_masks.shape[0]):
                b = budgets[c]
                if b <= 0 or b not in sched:
                    continue
                S, v_S = sched[b]
                gt = set(int(i) for i in np.where(gt_masks[c])[0])
                jac = len(gt & S) / len(gt | S)
                oracle.append({"candidate": c, "v_found": float(v_S),
                               "v_gt": float(game(
                                   sum(1 << p for p in gt)))})
                if np.isnan(best) or jac > best:
                    best, best_c = jac, c
        out["records"].append({
            "orig_index": orig, "label": label, "n_nodes": n,
            "gea": best, "candidate": best_c,
            "gt_candidates": int(gt_masks.shape[0]),
            "oracle": oracle, "evaluations": game.calls})
        if (done + 1) % 10 == 0:
            print(f"  {done + 1}/{len(entries)}", flush=True)
            out_file.write_text(json.dumps(out))

    g = [r["gea"] for r in out["records"]]
    out["gea_mean"] = float(np.nanmean(g))
    out_file.write_text(json.dumps(out))
    print(f"select-on-{arch} GEA {out['gea_mean']:.4f} over {len(g)}; "
          f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
