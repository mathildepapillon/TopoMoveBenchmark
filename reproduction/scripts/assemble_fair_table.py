"""Assemble the fair Arm-1 comparison table from the local campaigns.

Sources (all same TopoBench splits, full gt-positive test populations,
same scorer — toposhap.metrics.gea.graph_explanation_accuracy):
  results/molecule_gea/*_full*.json          TopoSHAP on k3/k6 GCCN
  results/graph_baselines/*_ran-gnn-pge.json random/GNNE/PGE on GCN+GIN
  results/graph_baselines/*_sub*.json        SubgraphX on GCN+GIN

Shard files are strided slices of one population: records are merged on
orig_index and the merge asserts completeness against the population
size recorded at training time (train_{ds}.jsonl n_test_gt_positive).

Comparisons are at node level (TopoSHAP cell attributions projected to
nodes — records' ``gea_node``; baselines are node-native). Writes
results/fair_table/summary.json and prints the table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GEA = REPO / "results" / "molecule_gea"
GB = REPO / "results" / "graph_baselines"

DATASETS = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai",
            "alkane_carbonyl"]
SEEDS = [42, 43, 44]


def population(ds: str) -> int:
    rows = [json.loads(x) for x in
            (GB / f"train_{ds}.jsonl").read_text().splitlines() if x]
    (n,) = {r["n_test_gt_positive"] for r in rows}
    return n


def merge(files, key):
    """Merge per-graph values across shard files; orig_index-deduped."""
    vals = {}
    for f in files:
        j = json.loads(Path(f).read_text())
        for r in j["records"]:
            v = key(r)
            prev = vals.setdefault(r["orig_index"], v)
            assert prev == v, (f, r["orig_index"], "conflicting shards")
    return vals


def toposhap_rows(ds: str, karm: str) -> dict[int, float]:
    tag = "_full" if karm == "k3" else f"_{karm}_full"
    per_seed = {}
    for seed in SEEDS:
        files = sorted(GEA.glob(
            f"{ds}_participation_p128_s{seed}{tag}*.json"))
        if karm == "k3":
            files = [f for f in files
                     if "_k6_" not in f.name and "_k11_" not in f.name]
        vals = merge(files, lambda r: r["gea_node"])
        assert len(vals) == population(ds), (ds, karm, seed, len(vals))
        per_seed[seed] = float(np.mean(list(vals.values())))
    return per_seed


def select_rows(ds: str, karm: str) -> dict[int, float]:
    """TopoSHAP-Select (coalition readout) rows, shard-merged."""
    sel = REPO / "results" / "molecule_select"
    per_seed = {}
    for seed in SEEDS:
        files = sorted(sel.glob(f"{ds}_select_s{seed}_{karm}_full*.json"))
        vals = merge(files, lambda r: r["gea_node"])
        assert len(vals) == population(ds), (ds, karm, seed, len(vals))
        per_seed[seed] = float(np.mean(list(vals.values())))
    return per_seed


def baseline_rows(ds: str, arch: str, method: str) -> dict[int, float]:
    per_seed = {}
    for seed in SEEDS:
        if method == "subgraphx":
            files = sorted(GB.glob(f"{ds}_{arch}_s{seed}_sub*.json"))
            files = [f for f in files if "_l" not in f.name.split(ds)[1]]
        else:
            files = [GB / f"{ds}_{arch}_s{seed}_ran-gnn-pge.json"]
        vals = merge(files, lambda r: r["methods"][method]["gea"])
        assert len(vals) == population(ds), (ds, arch, method, seed,
                                             len(vals))
        per_seed[seed] = float(np.mean(list(vals.values())))
    return per_seed


def main() -> None:
    out = {"scoring": "node GEA, Jaccard at gt size, max over candidates",
           "population": "full gt-positive test split (TopoBench splits)",
           "rows": {}}
    table = defaultdict(dict)
    karms = ["k3", "k6"]
    if list(GEA.glob("*_k11_full*.json")):
        karms.append("k11")
    sel_dir = REPO / "results" / "molecule_select"
    for ds in DATASETS:
        for karm in karms:
            try:
                table[ds][f"toposhap_{karm}"] = toposhap_rows(ds, karm)
            except AssertionError:
                # population not complete yet (fleet in flight): mark
                # pending rather than shipping a partial-population cell
                table[ds][f"toposhap_{karm}"] = None
        for karm in ("k6", "k11"):
            if not list(sel_dir.glob(f"*_select_*_{karm}_full*.json")):
                continue
            try:
                table[ds][f"select_{karm}"] = select_rows(ds, karm)
            except AssertionError:
                table[ds][f"select_{karm}"] = None
        for arch in ("gcn", "gin"):
            for meth in ("random", "gnnexplainer", "pgexplainer",
                         "subgraphx"):
                table[ds][f"{meth}_{arch}"] = baseline_rows(ds, arch, meth)
    for ds in DATASETS:
        out["rows"][ds] = {"n": population(ds)}
        for name, per_seed in table[ds].items():
            if per_seed is None:
                out["rows"][ds][name] = None
                continue
            v = list(per_seed.values())
            out["rows"][ds][name] = {
                "mean": float(np.mean(v)), "sd": float(np.std(v)),
                "per_seed": per_seed}

    names = list(next(iter(table.values())).keys())
    print(f"{'':22s}" + "".join(f"{d[:12]:>14s}" for d in DATASETS))
    print(f"{'n =':22s}" + "".join(
        f"{population(d):>14d}" for d in DATASETS))
    for name in names:
        cells = []
        for ds in DATASETS:
            r = out["rows"][ds].get(name)
            cells.append(f"{r['mean']:.3f}±{r['sd']:.3f}" if r
                         else "(pending)")
        print(f"{name:22s}" + "".join(f"{c:>14s}" for c in cells))

    dest = REPO / "results" / "fair_table"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", dest / "summary.json")


if __name__ == "__main__":
    main()
