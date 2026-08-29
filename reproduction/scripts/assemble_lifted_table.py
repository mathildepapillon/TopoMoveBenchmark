"""Assemble the lifted-benchmarks table: TopoSHAP on cell complexes.

The Part-2 companion to the fair rank-1 table: the GraphXAI molecule
benchmarks lifted to cell complexes (rings are rank-2 players), the
GCCN subject families (k = 3 recipe-selected, 6, 11 = full vocabulary),
and BOTH outputs of the method on every subject:

  attribution  per-cell Shapley, top-k at ground-truth size
               (results/molecule_gea/*_full*.json)
  explanation  greedy-prune coalition at ground-truth size
               (results/molecule_select/*_full*.json)

Reported at cell level (the native metric on the lifted complex: the
ground truth is lifted to cells, closure included) and node-projected
(comparable with the rank-1 table). Alignment ceiling per subject =
share of candidates where v(found) > v(gt) in the coalition runs.
Cell-level random floor from the attribution runs' per-graph controls.

Writes results/fair_table/lifted_summary.json and prints the table.
Incomplete populations print as (pending), never as partial numbers.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from assemble_fair_table import DATASETS, SEEDS, merge, population  # noqa: E402

GEA = REPO / "results" / "molecule_gea"
SEL = REPO / "results" / "molecule_select"
KARMS = ["k3", "k6", "k11", "k11s"]


def attr_files(ds: str, karm: str, seed: int):
    tag = "_full" if karm == "k3" else f"_{karm}_full"
    files = sorted(GEA.glob(f"{ds}_participation_p128_s{seed}{tag}*.json"))
    if karm == "k3":
        files = [f for f in files
                 if "_k6_" not in f.name and "_k11_" not in f.name]
    return files


def per_seed_means(ds, karm, files_fn, key):
    out = {}
    for seed in SEEDS:
        vals = merge(files_fn(ds, karm, seed), key)
        if len(vals) != population(ds):
            return None  # incomplete population: pending, never partial
        out[seed] = float(np.nanmean(list(vals.values())))
    return out


def sel_files(ds: str, karm: str, seed: int):
    return sorted(f for f in
                  SEL.glob(f"{ds}_select_s{seed}_{karm}_full*.json")
                  if "_lc" not in f.name)


def sel_files_lc(ds: str, karm: str, seed: int):
    """Lift-consistent semantics runs (feature-cache leak closed)."""
    return sorted(SEL.glob(f"{ds}_select_s{seed}_{karm}_full_lc*.json"))


def subject_accuracy(ds: str, karm: str):
    accs = []
    if karm == "k3":
        for seed in SEEDS:
            row = json.loads(
                (REPO / "results" / "anchored3_exact" /
                 f"anchored3_{ds}_seed{seed}.jsonl").read_text())
            accs.append(row["test_accuracy"])
    else:
        rows = [json.loads(x) for x in
                (REPO / "results" / f"molecule_top{karm}" /
                 f"top{karm}_{ds}.jsonl").read_text().splitlines() if x]
        accs = [r["test_accuracy"] for r in rows if r["seed"] in SEEDS]
    return float(np.mean(accs)), float(np.std(accs))


def ceiling(ds: str, karm: str):
    """Share of candidates with v(found) > v(gt), pooled over seeds."""
    beats = total = 0
    for seed in SEEDS:
        for f in sel_files(ds, karm, seed):
            for r in json.loads(f.read_text())["records"]:
                for o in r["oracle"]:
                    total += 1
                    beats += int(o["v_found"] > o["v_gt"])
    return (beats / total) if total else None


def main() -> None:
    out = {"scoring": "GEA, Jaccard at gt size, max over candidates",
           "population": "full gt-positive test split", "rows": {}}
    for ds in DATASETS:
        row = {"n": population(ds)}
        for karm in KARMS:
            acc = subject_accuracy(ds, karm)
            cell = {}
            for outp, files_fn, base in (
                    ("attribution", attr_files, GEA),
                    ("explanation", sel_files, SEL),
                    ("explanation_lc", sel_files_lc, SEL)):
                for level, key in (("cell", lambda r: r["gea"]),
                                   ("node", lambda r: r["gea_node"])):
                    ps = per_seed_means(ds, karm, files_fn, key)
                    cell[f"{outp}_{level}"] = (None if ps is None else {
                        "mean": float(np.mean(list(ps.values()))),
                        "sd": float(np.std(list(ps.values()))),
                        "per_seed": ps})
            rand = per_seed_means(ds, karm, attr_files,
                                  lambda r: r["random_gea_mean"])
            cell["random_cell"] = (None if rand is None else
                                   float(np.mean(list(rand.values()))))
            cell["accuracy"] = {"mean": acc[0], "sd": acc[1]}
            cell["ceiling_share"] = ceiling(ds, karm)
            row[karm] = cell
        out["rows"][ds] = row

    def fmt(c):
        return f"{c['mean']:.3f}±{c['sd']:.3f}" if c else "(pending)"

    for level in ("cell", "node"):
        print(f"\n== {level}-level GEA (lifted complexes, full "
              f"populations, 3 seeds) ==")
        print(f"{'':28s}" + "".join(f"{d[:12]:>14s}" for d in DATASETS))
        for karm in KARMS:
            for outp in ("attribution", "explanation", "explanation_lc"):
                cells = [fmt(out["rows"][ds][karm][f"{outp}_{level}"])
                         for ds in DATASETS]
                print(f"{karm:5s} {outp:22s}" +
                      "".join(f"{c:>14s}" for c in cells))
        if level == "cell":
            cells = []
            for ds in DATASETS:
                r = out["rows"][ds]["k6"]["random_cell"]
                cells.append(f"{r:.3f}" if r is not None else "(pending)")
            print(f"{'random floor (cell)':28s}" +
                  "".join(f"{c:>14s}" for c in cells))

    print("\n== subject accuracy / alignment ceiling "
          "(share of candidates with v(found) > v(gt)) ==")
    for karm in KARMS:
        cells = []
        for ds in DATASETS:
            a = out["rows"][ds][karm]["accuracy"]
            c = out["rows"][ds][karm]["ceiling_share"]
            cells.append(f"{a['mean']:.3f}/"
                         f"{c:.2f}" if c is not None else
                         f"{a['mean']:.3f}/--")
        print(f"{karm:28s}" + "".join(f"{c:>14s}" for c in cells))

    dest = REPO / "results" / "fair_table"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "lifted_summary.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", dest / "lifted_summary.json")


if __name__ == "__main__":
    main()
