"""Arm-1 fair baselines: graph explainers on the locally trained GNNs.

Runs the graph-XAI baseline suite under the SAME protocol as our
TopoSHAP GEA runs (experiments/molecule_participation_gea.py): same
TopoBench test split (asserted at training time by train_graph_gnns.py),
every ground-truth-positive test molecule, the same scorer
(toposhap.metrics.gea.graph_explanation_accuracy — Jaccard at
ground-truth size, max over GraphXAI candidates), and every explainer
targeting the true label class (the leg-1 convention).

Methods:
  random        seeded uniform scores through the identical scorer
                (mean over 32 draws per graph)
  gnnexplainer  PyG torch_geometric.explain GNNExplainer, node mask
  pgexplainer   PyG PGExplainer (trained on the training split),
                edge mask projected to nodes by max over incident edges
  subgraphx     GraphXAI SubgraphX (vendored at the pinned commit,
                scripts/apply_graphxai_patches.py), one MCTS run per
                distinct candidate size with max_nodes = that size

Usage:
    python run_graph_baselines.py DATASET ARCH SEED [METHODS_CSV]
Env: TOPOSHAP_GRAPH_BASELINE_LIMIT (smoke; 0/unset = full population).
Writes results/graph_baselines/{DATASET}_{ARCH}_s{SEED}.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "external" / "GraphXAI"))

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
from torch_geometric.data import Data  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import find_ckpt  # noqa: E402
from train_graph_gnns import compose_graph  # noqa: E402
from toposhap.metrics.gea import graph_explanation_accuracy  # noqa: E402

ALL_METHODS = ["random", "gnnexplainer", "pgexplainer", "subgraphx"]
RANDOM_DRAWS = 32


class GraphLogitAdapter(torch.nn.Module):
    """``model(x, edge_index, batch=...) -> logits`` for explainer APIs.

    Rebuilds a minimal TopoBench batch each call: the feature encoder
    mutates ``x_0`` in place, so the caller's tensors must never be the
    batch's storage.
    """

    def __init__(self, tb_model):
        super().__init__()
        self.tb = tb_model

    def forward(self, x, edge_index, batch=None, **_):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long,
                                device=x.device)
        data = Data(x=x, x_0=x, edge_index=edge_index, batch_0=batch)
        return self.tb.forward(data)["logits"]


def load_model(ds: str, arch: str, seed: int, device):
    rows = [json.loads(x) for x in
            (REPO / "results" / "graph_baselines" / f"train_{ds}.jsonl")
            .read_text().splitlines() if x]
    row = next(r for r in rows if r["arch"] == arch and r["seed"] == seed)
    run_dir = REPO / "results" / "graph_baselines" / "runs" / f"{ds}_{arch}_s{seed}"
    cfg = compose_graph(ds, seed, arch, run_dir)
    pipe = drv.build_pipeline(cfg)
    ckpt = find_ckpt(run_dir / "checkpoints", row.get("selected_epoch"))
    state = torch.load(str(ckpt), map_location="cpu",
                       weights_only=False)["state_dict"]
    pipe.model.load_state_dict(state, strict=True)
    pipe.model.to(device).eval()
    acc, _ = drv.accuracy_over_loader(
        pipe.model, pipe.datamodule.test_dataloader(), "test", device)
    print(f"{ds} {arch} s{seed}: test acc {acc:.4f} "
          f"(trained row: {row['test_accuracy']:.4f})", flush=True)
    if abs(acc - row["test_accuracy"]) > 0.02:
        raise SystemExit("checkpoint does not reproduce the trained test "
                         "accuracy — refusing")
    return cfg, pipe, row, float(acc)


def node_scores_gnnexplainer(explainer, x, edge_index, target, bvec):
    exp = explainer(x, edge_index, target=target, batch=bvec)
    return exp.node_mask.detach().view(-1).cpu().numpy()


def edge_to_node_scores(edge_mask, edge_index, n_nodes):
    """Node score = max over incident edge scores (GraphXAI convention)."""
    scores = np.zeros(n_nodes)
    em = edge_mask.detach().cpu().numpy()
    ei = edge_index.cpu().numpy()
    for e in range(ei.shape[1]):
        for v in (ei[0, e], ei[1, e]):
            scores[v] = max(scores[v], em[e])
    return scores


def main() -> None:
    ds, arch, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    methods = (sys.argv[4].split(",") if len(sys.argv) > 4 else ALL_METHODS)
    assert all(m in ALL_METHODS for m in methods), methods

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
    # TOPOSHAP_GRAPH_BASELINE_SHARD="i/k": strided slice, files suffixed,
    # merged back at analysis time (mirrors TOPOSHAP_GEA_SHARD).
    shard = os.environ.get("TOPOSHAP_GRAPH_BASELINE_SHARD", "")
    if shard:
        si, sk = (int(v) for v in shard.split("/"))
        entries = entries[si::sk]
    print(f"explaining {len(entries)} gt-positive test molecules"
          f"{f' (shard {shard})' if shard else ''} ({methods})", flush=True)

    from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

    model_config = dict(mode="multiclass_classification",
                        task_level="graph", return_type="raw")

    gnne = pge = sgx = None
    if "gnnexplainer" in methods:
        gnne = Explainer(
            model=adapter, algorithm=GNNExplainer(epochs=200),
            explanation_type="phenomenon", node_mask_type="object",
            edge_mask_type="object", model_config=model_config)
    if "pgexplainer" in methods:
        algo = PGExplainer(epochs=30, lr=0.003).to(device)
        pge = Explainer(
            model=adapter, algorithm=algo,
            explanation_type="phenomenon", edge_mask_type="object",
            model_config=model_config)
        train_graphs = pipe.datamodule.dataset_train.data_lst
        t0 = time.perf_counter()
        for epoch in range(algo.epochs):
            for d in train_graphs:
                x = d.x.to(device)
                ei = d.edge_index.to(device)
                bvec = torch.zeros(x.shape[0], dtype=torch.long,
                                   device=device)
                y = torch.as_tensor(
                    np.asarray(d.y).reshape(-1)[:1], device=device)
                algo.train(epoch, adapter, x, ei, target=y, batch=bvec)
        print(f"pgexplainer trained on {len(train_graphs)} graphs "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    if "subgraphx" in methods:
        from graphxai.explainers import SubgraphX

        sgx = SubgraphX(adapter)

    out = {"dataset": ds, "arch": arch, "seed": seed,
           "test_accuracy": acc, "n_params": row["n_params"],
           "methods": methods, "shard": shard,
           "n_population": len(entries), "records": []}
    out_dir = REPO / "results" / "graph_baselines"
    suffix = f"_l{limit}" if limit > 0 else ""
    if methods != ALL_METHODS:
        suffix += "_" + "-".join(m[:3] for m in methods)
    if shard:
        suffix += f"_sh{shard.replace('/', 'of')}"
    out_file = out_dir / f"{ds}_{arch}_s{seed}{suffix}.json"

    for done, (orig, pos) in enumerate(entries):
        d = test.data_lst[pos]
        x = d.x.to(device)
        ei = d.edge_index.to(device)
        n_nodes = int(x.shape[0])
        bvec = torch.zeros(n_nodes, dtype=torch.long, device=device)
        label = int(np.asarray(d.y).reshape(-1)[0])
        target = torch.tensor([label], device=device)
        gt_masks = source.gt_candidates[orig]["node"].cpu().numpy() > 0
        rec = {"orig_index": orig, "label": label, "n_nodes": n_nodes,
               "gt_candidates": int(gt_masks.shape[0]), "methods": {}}

        for meth in methods:
            t0 = time.perf_counter()
            if meth == "random":
                rng = np.random.default_rng(orig)
                geas = [graph_explanation_accuracy(
                    rng.random(n_nodes), gt_masks)["gea"]
                    for _ in range(RANDOM_DRAWS)]
                res = {"gea": float(np.mean(geas)),
                       "gea_sd_draws": float(np.std(geas))}
            elif meth == "gnnexplainer":
                scores = node_scores_gnnexplainer(gnne, x, ei, target, bvec)
                res = graph_explanation_accuracy(scores, gt_masks)
            elif meth == "pgexplainer":
                exp = pge(x, ei, target=target, batch=bvec)
                scores = edge_to_node_scores(exp.edge_mask, ei, n_nodes)
                res = graph_explanation_accuracy(scores, gt_masks)
            elif meth == "subgraphx":
                sizes = sorted({int(m.sum()) for m in gt_masks
                                if m.sum() > 0})
                best, mask_sizes = None, []
                for s in sizes:
                    exp = sgx.get_explanation_graph(
                        x, ei, label=label, max_nodes=s,
                        forward_kwargs={"batch": bvec})
                    scores = exp.node_imp.detach().cpu().numpy()
                    mask_sizes.append(int((scores > 0).sum()))
                    same = gt_masks[[i for i in range(gt_masks.shape[0])
                                     if int(gt_masks[i].sum()) == s]]
                    r = graph_explanation_accuracy(scores, same)
                    if best is None or r["gea"] > best["gea"]:
                        best = r
                res = dict(best)
                res["mask_sizes"] = mask_sizes
            res["seconds"] = round(time.perf_counter() - t0, 3)
            rec["methods"][meth] = res
        out["records"].append(rec)
        if (done + 1) % 10 == 0:
            print(f"  {done + 1}/{len(entries)}", flush=True)
            out_file.write_text(json.dumps(out))

    out["summary"] = {}
    for meth in methods:
        geas = [r["methods"][meth]["gea"] for r in out["records"]]
        out["summary"][meth] = {"gea_mean": float(np.mean(geas)),
                                "gea_sd_graphs": float(np.std(geas))}
        print(f"{meth}: GEA {out['summary'][meth]['gea_mean']:.4f} "
              f"over {len(geas)} graphs", flush=True)
    out_file.write_text(json.dumps(out))
    print("wrote", out_file, flush=True)


if __name__ == "__main__":
    main()
