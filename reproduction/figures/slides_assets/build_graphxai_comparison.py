"""GraphXAI-benchmark comparison figure for the slide deck.

One row of four panels (Benzene, Fluoride-Carbonyl, Mutagenicity,
Alkane-Carbonyl), GEA (node-projected Jaccard) on a linear 0-1 y-axis
uniform across panels.  Every our-side number is read directly from the
frozen store (data/frozen/attribution4/all_tables.json, table
``baseline_join``) at build time and cross-checked against hard-coded
expected values; a mismatch aborts the build.  FORGE rows are literature
values (FORGE paper v5, 10 seeds, explaining flat GNNs) and are marked as
such -- they have no local provenance.

Outputs (deterministic):
  figures/slides_assets/graphxai_comparison.png   (2400 px wide, 2x for 1200)
  figures/slides_assets/graphxai_comparison.svg
  figures/slides_assets/graphxai_numbers.json     (every plotted value + locator)

Known deck discrepancy (intentionally corrected here): the deck's
"TopoSHAP explaining the better flat GNN" Alkane-Carbonyl cell says 0.343,
which traces to the non-primary ``motif_matched_correct`` selection.  The
primary campaign selection (``smallest_gt_positive``, the one used for every
other cell in that row and carried by ``baseline_join``) gives 0.336, which
is what this figure plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen" / "attribution4" / "all_tables.json"
OUT_DIR = Path(__file__).resolve().parent

DATASETS = ["Benzene", "FluorideCarbonyl", "Mutagenicity", "AlkaneCarbonyl"]
TITLES = {
    "Benzene": "Benzene",
    "FluorideCarbonyl": "Fluoride-Carbonyl",
    "Mutagenicity": "Mutagenicity",
    "AlkaneCarbonyl": "Alkane-Carbonyl",
}
GCCN_MODEL = "topotune_nonlinear_exact_plus"
FLAT_MODELS = ("gcn", "gin")
BASELINE_EXPLAINERS = ("gnnexplainer", "pgexplainer", "subgraphx")

# Literature rows -- FORGE paper (arXiv:2406.03253 / AAAI 2025), arXiv v5,
# 10 seeds, explaining flat GNNs.  cite-only per
# data/frozen/baselines2/manifest.json["forge"]; NOT verifiable locally.
FORGE_PLAIN = {"Benzene": 0.456, "FluorideCarbonyl": 0.233,
               "Mutagenicity": 0.380, "AlkaneCarbonyl": 0.114}
FORGE_BOOST = {"Benzene": 0.772, "FluorideCarbonyl": 0.441,
               "Mutagenicity": 0.339, "AlkaneCarbonyl": 0.304}

# Guard rails: expected frozen values (3 dp).  Build aborts on drift.
EXPECT = {
    ("ours_gccn", "Benzene"): 0.725, ("ours_gccn", "FluorideCarbonyl"): 0.637,
    ("ours_gccn", "Mutagenicity"): 0.744, ("ours_gccn", "AlkaneCarbonyl"): 0.236,
    ("ours_flat", "Benzene"): 0.784, ("ours_flat", "FluorideCarbonyl"): 0.515,
    ("ours_flat", "Mutagenicity"): 0.550, ("ours_flat", "AlkaneCarbonyl"): 0.336,
    ("baseline", "Benzene"): 0.752, ("baseline", "FluorideCarbonyl"): 0.483,
    ("baseline", "Mutagenicity"): 0.394, ("baseline", "AlkaneCarbonyl"): 0.236,
    ("random", "Benzene"): 0.238, ("random", "FluorideCarbonyl"): 0.210,
    ("random", "Mutagenicity"): 0.129, ("random", "AlkaneCarbonyl"): 0.142,
}

# Palette (validated with the dataviz six-checks script; CVD + normal-vision
# separation PASS; the muted baseline gray is a deliberate design choice and
# identity is never color-alone: fixed bar order + per-bar name tags).
C_OURS_GCCN = "#24409e"
C_OURS_FLAT = "#5c85e8"
C_BASELINE = "#9aa0a8"
C_FORGE_PLAIN = "#eda54a"
C_FORGE_BOOST = "#bf5b04"
C_RANDOM = "#555555"
INK = "#1a1a1a"
MUTED = "#5a5f66"


def load_frozen():
    tables = json.loads(FROZEN.read_text())
    rows = tables["baseline_join"]["rows"]

    def campaign(dataset, model):
        (r,) = [r for r in rows if r["source"] == "campaign"
                and r["dataset"] == dataset and r["model"] == model]
        return r

    def suite(dataset, explainer, model):
        (r,) = [r for r in rows if r["source"] == "baseline_suite"
                and r["dataset"] == dataset and r["explainer"] == explainer
                and r["model"] == model]
        return r

    data = {}
    for ds in DATASETS:
        loc = "data/frozen/attribution4/all_tables.json#baseline_join.rows"
        g = campaign(ds, GCCN_MODEL)
        flat_rows = {m: campaign(ds, m) for m in FLAT_MODELS}
        flat_best = max(FLAT_MODELS, key=lambda m: flat_rows[m]["gea_node_mean"])
        f = flat_rows[flat_best]
        base_rows = {(e, m): suite(ds, e, m)
                     for e in BASELINE_EXPLAINERS for m in FLAT_MODELS}
        (be, bm) = max(base_rows, key=lambda k: base_rows[k]["gea_node_mean"])
        b = base_rows[(be, bm)]
        rnd = suite(ds, "random", "gcn")
        data[ds] = {
            "ours_gccn": {
                "gea": g["gea_node_mean"], "sd": g["gea_node_sd"],
                "seeds": 5, "model": GCCN_MODEL,
                "locator": f"{loc}[source=campaign, dataset={ds}, "
                           f"model={GCCN_MODEL}].gea_node_mean/.gea_node_sd",
            },
            "ours_flat": {
                "gea": f["gea_node_mean"], "sd": f["gea_node_sd"],
                "seeds": 5, "model": flat_best,
                "locator": f"{loc}[source=campaign, dataset={ds}, "
                           f"model={flat_best}].gea_node_mean/.gea_node_sd "
                           f"(max over gcn/gin)",
            },
            "baseline": {
                "gea": b["gea_node_mean"], "explainer": be, "model": bm,
                "n_graphs": b["n_graphs"],
                "locator": f"{loc}[source=baseline_suite, dataset={ds}, "
                           f"explainer={be}, model={bm}].gea_node_mean "
                           f"(max over explainer x model; single training "
                           f"run, sd is per-graph spread, not per-seed)",
            },
            "random": {
                "gea": rnd["gea_node_mean"],
                "locator": f"{loc}[source=baseline_suite, dataset={ds}, "
                           f"explainer=random].gea_node_mean",
            },
        }
    # guard rails
    for (series, ds), want in EXPECT.items():
        got = round(data[ds][series]["gea"], 3)
        assert got == want, f"frozen drift: {series}/{ds} = {got}, expected {want}"
    return data


def write_numbers_json(data):
    doc = {
        "figure": "graphxai_comparison.png / .svg",
        "metric": "GEA -- node-projected ground-truth explanation agreement "
                  "(Jaccard, higher better)",
        "built_from": "data/frozen/attribution4/all_tables.json (frozen; "
                      "identical to data/_staging_t1/table1/exp4_tables/"
                      "all_tables.json), experiment exp_01kzt8hvm7eqytdzfejz55a1h5",
        "conventions": {
            "ours_rows": "campaign selection (smallest_gt_positive), "
                         "per-seed mean, sd over 5 seeds",
            "baseline_rows": "baseline suite (data/frozen/baselines2, "
                             "exp_01kzs9avf8e18vw7d6ptc94rqt): best of "
                             "GNNExplainer/PGExplainer/SubgraphX on GCN/GIN, "
                             "graphxai_default and selected_on_train variants "
                             "pooled, single training run each -- no per-seed "
                             "error bars exist",
            "forge_rows": "LITERATURE values, FORGE paper v5 "
                          "(arXiv:2406.03253, AAAI 2025), 10 seeds, explaining "
                          "flat GNNs; cite-only per data/frozen/baselines2/"
                          "manifest.json#forge -- NOT verifiable against local "
                          "frozen data",
        },
        "deck_discrepancies": [
            {
                "cell": "TopoSHAP explaining the better flat GNN, "
                        "Alkane-Carbonyl",
                "deck_value": 0.343,
                "frozen_value": 0.336,
                "detail": "deck's 0.343 traces to motif_matched_correct "
                          "(gcn, gea_node_mean=0.3428), a non-primary "
                          "selection; the campaign/primary selection "
                          "(smallest_gt_positive, used for every other cell "
                          "in the row and carried by baseline_join) gives "
                          "0.3356. Figure plots the frozen primary value "
                          "0.336.",
            },
            {
                "cell": "FORGE literature rows vs frozen forge_reference_gea",
                "detail": "data/frozen/baselines2/summary.json carries a "
                          "forge_reference_gea column with per-explainer "
                          "values (e.g. Benzene subgraphx 0.622) that match "
                          "neither deck FORGE row; the deck itself notes "
                          "earlier arXiv revisions differ. Treated as "
                          "literature, plotted as deck-stated v5 values.",
            },
        ],
        "cross_references": {
            "claims_ledger": "data/frozen/reference/synthesis_report/"
                             "claims_ledger.json (sgx_ours_*, sgx_ref_*, "
                             "benzene_chance)",
            "macros": "paper/results_macros.tex (\\ResGEAOurs*, "
                      "\\ResGEASubgraphX*, \\ResGEARandomBenzene)",
        },
        "datasets": {},
    }
    for ds in DATASETS:
        d = data[ds]
        doc["datasets"][TITLES[ds]] = {
            "toposhap_on_gccn": {
                "gea": round(d["ours_gccn"]["gea"], 6),
                "sd_over_5_seeds": round(d["ours_gccn"]["sd"], 6),
                "explained_model": d["ours_gccn"]["model"],
                "provenance": d["ours_gccn"]["locator"],
            },
            "toposhap_on_best_flat_gnn": {
                "gea": round(d["ours_flat"]["gea"], 6),
                "sd_over_5_seeds": round(d["ours_flat"]["sd"], 6),
                "explained_model": d["ours_flat"]["model"],
                "provenance": d["ours_flat"]["locator"],
            },
            "best_flat_baseline": {
                "gea": round(d["baseline"]["gea"], 6),
                "explainer": d["baseline"]["explainer"],
                "explained_model": d["baseline"]["model"],
                "provenance": d["baseline"]["locator"],
            },
            "random_control": {
                "gea": round(d["random"]["gea"], 6),
                "provenance": d["random"]["locator"],
            },
            "forge_best_plain_reported": {
                "gea": FORGE_PLAIN[ds],
                "provenance": "LITERATURE: FORGE paper v5 (arXiv:2406.03253), "
                              "10 seeds, explaining flat GNNs -- no local "
                              "frozen source",
            },
            "forge_best_boosted_reported": {
                "gea": FORGE_BOOST[ds],
                "provenance": "LITERATURE: FORGE paper v5 (arXiv:2406.03253), "
                              "10 seeds, explaining flat GNNs -- no local "
                              "frozen source",
            },
        }
    out = OUT_DIR / "graphxai_numbers.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return out


BAR_TAGS = ["TopoSHAP → GCCN", "TopoSHAP → flat GNN", "best flat baseline",
            "FORGE plain*", "FORGE boosted*"]


def draw(data):
    plt.rcParams.update({
        "svg.hashsalt": "toposhap-graphxai",
        "font.size": 12.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.edgecolor": "#c9ccd1",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "path.simplify": False,
    })
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 5.1), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.992, top=0.885, bottom=0.265,
                        wspace=0.16)

    for i, (ds, ax) in enumerate(zip(DATASETS, axes)):
        d = data[ds]
        vals = [d["ours_gccn"]["gea"], d["ours_flat"]["gea"],
                d["baseline"]["gea"], FORGE_PLAIN[ds], FORGE_BOOST[ds]]
        sds = [d["ours_gccn"]["sd"], d["ours_flat"]["sd"], None, None, None]
        colors = [C_OURS_GCCN, C_OURS_FLAT, C_BASELINE,
                  C_FORGE_PLAIN, C_FORGE_BOOST]
        xs = range(5)

        # random floor first (behind the bars)
        rnd = d["random"]["gea"]
        ax.axhline(rnd, color=C_RANDOM, lw=1.3, ls=(0, (4, 3)), alpha=0.65,
                   zorder=1)
        if i == 0:
            # left margin, clear of the bars (a white bbox over a bar
            # cuts a notch out of it)
            ax.text(-0.55, rnd + 0.03, "random", ha="left", va="bottom",
                    fontsize=9.5, color="#3d4147", style="italic", zorder=6)

        for x, v, sd, c in zip(xs, vals, sds, colors):
            hatch = "//" if x >= 3 else None
            ax.bar(x, v, width=0.66, color=c, zorder=3, hatch=hatch,
                   edgecolor="white" if hatch else c,
                   linewidth=0.9 if hatch else 0.0)
            top = v
            if sd is not None:
                ax.errorbar(x, v, yerr=sd, fmt="none", ecolor=INK,
                            elinewidth=1.5, capsize=4.5, capthick=1.5,
                            zorder=4)
                top = v + sd
            if x == 1:  # which flat GNN the max picked
                ax.text(x, 0.028, d["ours_flat"]["model"].upper(),
                        ha="center", va="bottom", zorder=5, fontsize=8.3,
                        fontweight="bold", color="white")
            is_winner = v == max(vals)
            ax.text(x, top + 0.025, f"{v:.2f}",
                    ha="center", va="bottom", zorder=5,
                    fontsize=13.5 if is_winner else 11,
                    fontweight="bold" if is_winner else "normal",
                    color=INK if is_winner else MUTED)

        ax.set_title(TITLES[ds], fontsize=16.5, fontweight="bold", pad=12,
                     color=INK)
        ax.set_xlim(-0.62, 4.62)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines["left"].set_bounds(0, 1.0)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(BAR_TAGS, fontsize=10.2, rotation=33,
                           ha="right", rotation_mode="anchor")
        tag_cols = [C_OURS_GCCN, "#3f66c9", "#63686f", "#b07425", "#9c4a03"]
        for tick, c in zip(ax.get_xticklabels(), tag_cols):
            tick.set_color(c)
            tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.grid(axis="y", color="#e3e5e8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        if i == 0:
            ax.set_ylabel("GEA (Jaccard)  —  higher is better",
                          fontsize=12.5)
        if ds == "Benzene":
            # literature annotation over the two FORGE bars
            ax.plot([2.72, 4.33], [0.9, 0.9], color=MUTED, lw=1.0, zorder=5)
            for xx in (2.72, 4.33):
                ax.plot([xx, xx], [0.873, 0.9], color=MUTED, lw=1.0, zorder=5)
            ax.text(3.52, 0.92, "reported (10 seeds, v5)", ha="center",
                    va="bottom", fontsize=9.5, style="italic", color=MUTED,
                    zorder=6)
        if ds == "AlkaneCarbonyl":
            ax.text(2.0, 1.19, "hard for every method:\ngaps ≪ one seed-sd",
                    ha="center", va="top", fontsize=10, style="italic",
                    color=MUTED, zorder=6)

    fig.text(0.085, 0.052,
             "Ours (blues): TopoSHAP, mean ± sd over 5 seeds, frozen data.   "
             "Dashed line: random-explainer control.",
             fontsize=9.0, color=MUTED, ha="left", va="bottom")
    fig.text(0.085, 0.016,
             "Gray: best of GNNExplainer/PGExplainer/SubgraphX on GCN/GIN "
             "(single run).   *FORGE (hatched): means as published (v5 "
             "Table 2, 10 seeds, flat GNNs); no code release, and their "
             "seed spreads are in an appendix absent from the arXiv record.",
             fontsize=9.0, color=MUTED, ha="left", va="bottom")

    png = OUT_DIR / "graphxai_comparison.png"
    svg = OUT_DIR / "graphxai_comparison.svg"
    fig.savefig(png, dpi=200, facecolor="white",
                metadata={"Software": "toposhap figure builder"})
    fig.savefig(svg, facecolor="white",
                metadata={"Date": None, "Creator": "toposhap figure builder"})
    plt.close(fig)
    return png, svg


def main():
    data = load_frozen()
    nums = write_numbers_json(data)
    png, svg = draw(data)
    print("wrote", nums)
    print("wrote", png)
    print("wrote", svg)


if __name__ == "__main__":
    main()
