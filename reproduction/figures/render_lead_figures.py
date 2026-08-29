"""Render the two Results-II lead figures (canonical settings).

Marker policy (researcher-set, 2026-08-28): one dark marker per family
(OURS GCCN = the selection arm; OURS HOPSE = the selection arm in the
HOPSE family), one pale sweep marker per family, no menu, no fixed-k
variants, no reference lines, no per-seed circles.

For alkane_carbonyl, MUTAG, and NCI109 the GCCN sweep marker is
assembled from the frozen 64-config retraining enumeration
(data/frozen/pooled.json): y = best coalition mean with the frozen
seed-sd, x = estimated cost (64 runs, per-run FLOPs scaled from NCI1's
measured sweep by graph count; flops_estimated=True). The caption
declares the estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "figures"))

import build_lead_figure as blf  # noqa: E402

XAI = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai",
       "alkane_carbonyl"]
#: TopoBench figure: complex-level tasks on the top row (MANTRA is a
#: TopoBench dataset like the others), node-level tasks on the bottom
BENCH_ROWS = [["NCI1", "MUTAG", "NCI109", "mantra_orientation"],
              ["cocitation_cora", "cocitation_citeseer",
               "cocitation_pubmed"]]
BENCH = [d for row in BENCH_ROWS for d in row]
HIDE = {"menu", "k2", "k6", "ladder1", "ladder2", "hopse_k3", "seeds",
        "ceiling"}

#: graphs per dataset, for the per-run FLOPs scaling of estimated costs
N_GRAPHS = {"NCI1": 4110, "NCI109": 4127, "MUTAG": 188,
            "alkane_carbonyl": 1125}


def enumeration_rows(baselines: pd.DataFrame) -> pd.DataFrame:
    pooled = json.loads(
        (REPO / "data" / "frozen" / "pooled.json").read_text())
    retr = pooled["retraining_pooled"]
    nci1 = baselines[(baselines.dataset == "NCI1")
                     & (baselines.baseline == "gccn_sweep_best")]
    per_cfg_nci1 = float(nci1.flops.iloc[0]) / float(nci1.n_configs.iloc[0])
    rows = []
    for ds in ("alkane_carbonyl", "MUTAG", "NCI109"):
        d = retr[ds]
        best = max(d["values_mean"])
        sd = float(d["tolerance"]["seed_sd_median_over_coalitions"])
        flops = 64 * per_cfg_nci1 * N_GRAPHS[ds] / N_GRAPHS["NCI1"]
        for y in (best - sd, best, best + sd):
            rows.append(dict(dataset=ds, baseline="gccn_sweep_best",
                             n_configs=64, flops=flops,
                             wall_seconds=None, flops_estimated=True,
                             seed=-1, test_accuracy=y,
                             val_accuracy=None, winner="enum_best"))
    return pd.DataFrame(rows)


def main() -> None:
    recipe, menu, ceilings, demo, autok, baselines = blf.load_frozen(
        "anchored")
    if autok is not None and "selector" in autok.columns:
        autok = autok[autok.selector == "autok_backward"]
    baselines = baselines[baselines.baseline != "originals_best"]
    baselines = pd.concat([baselines, enumeration_rows(baselines)],
                          ignore_index=True)
    blf.build(recipe, menu, ceilings, demo, xcol="flops",
              selector="anchored", autok=autok, baselines=baselines,
              hide=HIDE, only=XAI, stem_name="lead_xai")
    blf.build(recipe, menu, ceilings, demo, xcol="flops",
              selector="anchored", autok=autok, baselines=baselines,
              hide=HIDE, rows=BENCH_ROWS, stem_name="lead_bench")
    print("rendered lead_xai + lead_bench")


if __name__ == "__main__":
    main()
