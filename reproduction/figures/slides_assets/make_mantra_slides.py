#!/usr/bin/env python
"""Build the two MANTRA story figures for the slide deck.

Every plotted number is verified against frozen data before drawing; any
mismatch raises. Outputs (in figures/slides_assets/):
  mantra_finding.png/.svg   -- (A) message-passing skill is a size artifact
  mantra_fix.png/.svg       -- (B) signed Hodge spectra close the gap
  mantra_numbers.json       -- every plotted value + provenance locator

Run:  .venv/bin/python figures/slides_assets/make_mantra_slides.py
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["svg.hashsalt"] = "mantra-slides"  # deterministic SVG ids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "slides_assets"
FROZEN = REPO / "data" / "frozen"

# ---------------------------------------------------------------- palette
# Reference dataviz palette (light mode), committed single-look for slides.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1 -- "all test" condition
ORANGE = "#eb6834"    # categorical slot 2 -- "size-matched" condition
VIOLET = "#4a3aa7"    # categorical slot 7 -- non-orientable class (inset)
GRAY_SERIES = "#898781"  # de-emphasis series  -- orientable class (inset)
ORD_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue 250/450/650

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 1.0,
    "svg.fonttype": "none",
})

TOL = 5e-4

def check(name, got, want, tol=TOL):
    if abs(got - want) > tol:
        raise SystemExit(
            f"FROZEN-DATA MISMATCH for {name}: computed {got!r}, "
            f"briefed {want!r}. Frozen value wins -- fix the brief/figure.")
    return got


def auroc(scores, labels):
    """Rank-based AUROC with tie handling (label 1 = positive)."""
    order = sorted(zip(scores, labels))
    vals = [v for v, _ in order]
    labs = [l for _, l in order]
    n = len(vals)
    npos = sum(labs)
    nneg = n - npos
    rsum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rsum += avg * sum(labs[i:j])
        i = j
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


# ================================================================ verify
attr_path = FROZEN / "mantra_attr_fig.json"
attr = json.loads(attr_path.read_text())
assert attr["seed"] == 43
assert attr["orientable_class"] == 0  # label 1 = NON-orientable; p = p(non-or)

table = attr["models"]["menu"]["test_table"]  # V,E,F,label identical across models
labels = [r["label"] for r in table]
n_test = len(table)
n_nonor = sum(labels)
n_or = n_test - n_nonor
assert (n_test, n_nonor, n_or) == (10785, 9930, 855)

mean_F_nonor = statistics.mean(r["F"] for r in table if r["label"] == 1)
mean_F_or = statistics.mean(r["F"] for r in table if r["label"] == 0)
check("mean F non-orientable", mean_F_nonor, 23.1633, 5e-3)
check("mean F orientable", mean_F_or, 20.4795, 5e-3)
count_only_auroc = check("count-only AUROC (F)",
                         auroc([r["F"] for r in table], labels), 0.7839)

# count-conditional AUROC recomputed within identical-(V,E,F) groups
groups = defaultdict(list)
for r in table:
    groups[(r["V"], r["E"], r["F"])].append(r)
mixed = {g: rows for g, rows in groups.items()
         if 0 < sum(x["label"] for x in rows) < len(rows)}
n_matched_complexes = sum(len(rows) for rows in mixed.values())
n_matched_pairs = sum(
    sum(x["label"] for x in rows) * (len(rows) - sum(x["label"] for x in rows))
    for rows in mixed.values())
assert n_matched_complexes == 5645 and n_matched_pairs == 1318027
assert len(mixed) == 4

cc = {}
for m in ("menu", "gccn", "hopse"):
    tab = attr["models"][m]["test_table"]
    by_g = defaultdict(list)
    for r in tab:
        by_g[(r["V"], r["E"], r["F"])].append(r)
    num = den = 0.0
    for rows in by_g.values():
        pos = [r["p"] for r in rows if r["label"] == 1]
        neg = [r["p"] for r in rows if r["label"] == 0]
        if pos and neg:
            num += sum(1.0 if a > b else (0.5 if a == b else 0.0)
                       for a in pos for b in neg)
            den += len(pos) * len(neg)
    cc[m] = check(f"cc-AUROC {m} (recomputed vs stored)", num / den,
                  attr["models"][m]["count_conditional_auroc"], 1e-9)
check("cc-AUROC menu", cc["menu"], 0.5000)
check("cc-AUROC gccn", cc["gccn"], 0.5056)
check("cc-AUROC hopse", cc["hopse"], 0.7909)

# seed-43 test AUROC (context / cross-check against parquets)
s43 = {m: attr["models"][m]["test_auroc"] for m in ("menu", "gccn", "hopse")}
check("seed-43 test AUROC menu", s43["menu"], 0.5209)
check("seed-43 test AUROC gccn", s43["gccn"], 0.7854)
check("seed-43 test AUROC hopse", s43["hopse"], 0.9260)

# --- 3-seed values from frozen parquets (per-seed val-ACCURACY selection) ---
menu_df = pd.read_parquet(FROZEN / "gccn_menu_mantra.parquet")
menu_pick = menu_df.loc[menu_df.groupby("seed")["val_accuracy"].idxmax()]
assert set(menu_pick["menu_name"]) == {"CWN"}  # selection lands on CWN each seed
menu_auroc = menu_pick["test_auroc"]
check("menu 3-seed AUROC mean", menu_auroc.mean(), 0.5209)
check("menu 3-seed AUROC sd", menu_auroc.std(ddof=1), 0.0000)
check("menu seed-43 == attr-fig menu", float(
    menu_pick.loc[menu_pick["seed"] == 43, "test_auroc"].iloc[0]), s43["menu"], 5e-4)

lad = pd.read_parquet(FROZEN / "ladder_b2_mantra.parquet")  # 1 selected row/seed
gccn_auroc = lad["test_auroc"]
check("gccn 3-seed AUROC mean", gccn_auroc.mean(), 0.7622)
check("gccn 3-seed AUROC sd", gccn_auroc.std(ddof=1), 0.0533)
# parquet logs 0.785413 vs attr-fig re-evaluation 0.785400 (same ckpt,
# re-scored at attribution time) -- sub-1e-4, both round to 0.7854
check("gccn seed-43 == attr-fig gccn", float(
    lad.loc[lad["seed"] == 43, "test_auroc"].iloc[0]), s43["gccn"], 5e-4)

sweep = pd.read_parquet(FROZEN / "hopse_sweep_mantra.parquet")
hp = sweep.loc[sweep.groupby("seed")["val_accuracy"].idxmax()]
hopse_auroc = hp["test_auroc"]
hopse_bal = hp["test_balanced_accuracy"]
check("hopse 3-seed AUROC mean", hopse_auroc.mean(), 0.9144)
check("hopse 3-seed AUROC sd", hopse_auroc.std(ddof=1), 0.0144)
check("hopse 3-seed balanced mean", hopse_bal.mean(), 0.7916)
check("hopse 3-seed balanced sd", hopse_bal.std(ddof=1), 0.0088)
check("hopse seed-43 == attr-fig hopse", float(
    hp.loc[hp["seed"] == 43, "test_auroc"].iloc[0]), s43["hopse"], 5e-4)

# --- rescue ladder (balanced accuracy, 3-seed means) from the claims ledger ---
ledger_path = FROZEN / "reference" / "synthesis_report" / "claims_ledger.json"
ledger = {c["id"]: c for c in json.loads(ledger_path.read_text())["claims"]}
bal_count = check("count_ceiling", ledger["count_ceiling"]["value"], 0.5054)
bal_struct = check("orient_struct", ledger["orient_struct"]["value"], 0.5407)
bal_spec = check("orient_spec", ledger["orient_spec"]["value"], 0.8764)

print("all frozen-data checks passed")

# binned triangle-count distribution for the fig-A inset (F is even; width-4
# bins remove the parity sawtooth)
BINS = [(8, 10), (12, 14), (16, 18), (20, 22), (24, 26), (28, 30)]
bin_centers = np.array([(a + b) / 2 for a, b in BINS])
sh_or = np.array([sum(1 for r in table if r["label"] == 0 and a <= r["F"] <= b)
                  for a, b in BINS]) / n_or
sh_no = np.array([sum(1 for r in table if r["label"] == 1 and a <= r["F"] <= b)
                  for a, b in BINS]) / n_nonor

# ================================================================ numbers.json
attr_rel = "data/frozen/mantra_attr_fig.json"
led_rel = "data/frozen/reference/synthesis_report/claims_ledger.json"
numbers = {
    "figure_A_mantra_finding": {
        "test_auroc_3seed": {
            "menu": {"mean": round(float(menu_auroc.mean()), 4),
                     "sd": round(float(menu_auroc.std(ddof=1)), 4),
                     "per_seed": [round(float(v), 4) for v in menu_auroc],
                     "source": "data/frozen/gccn_menu_mantra.parquet",
                     "locator": ("per-seed argmax val_accuracy over menu entries "
                                 "(selects CWN, mask 26, every seed); column test_auroc")},
            "gccn": {"mean": round(float(gccn_auroc.mean()), 4),
                     "sd": round(float(gccn_auroc.std(ddof=1)), 4),
                     "per_seed": [round(float(v), 4) for v in gccn_auroc],
                     "source": "data/frozen/ladder_b2_mantra.parquet",
                     "locator": "one validation-selected rung per seed; column test_auroc"},
            "hopse": {"mean": round(float(hopse_auroc.mean()), 4),
                      "sd": round(float(hopse_auroc.std(ddof=1)), 4),
                      "per_seed": [round(float(v), 4) for v in hopse_auroc],
                      "source": "data/frozen/hopse_sweep_mantra.parquet",
                      "locator": "per-seed argmax val_accuracy over 12 configs; column test_auroc"},
            "note": "seeds 42/43/44; selection rule reproduces deck values 0.52 / 0.76±0.05 / 0.91±0.01",
        },
        "count_conditional_auroc_seed43": {
            "menu": round(cc["menu"], 4), "gccn": round(cc["gccn"], 4),
            "hopse": round(cc["hopse"], 4),
            "source": attr_rel,
            "locator": ("models/<m>/count_conditional_auroc; recomputed exactly from "
                        "models/<m>/test_table within identical-(V,E,F) groups"),
            "matched_groups": 4, "matched_complexes": 5645,
            "matched_cross_class_pairs": 1318027,
        },
        "seed43_test_auroc_crosscheck": {
            "menu": round(s43["menu"], 4), "gccn": round(s43["gccn"], 4),
            "hopse": round(s43["hopse"], 4),
            "source": attr_rel, "locator": "models/<m>/test_auroc",
        },
        "count_only_auroc": {
            "value": round(count_only_auroc, 4), "source": attr_rel,
            "locator": ("AUROC of triangle count F as score for label==1 over "
                        "models/menu/test_table (E gives the same 0.7839; E=3F/2)"),
        },
        "mean_triangles": {
            "non_orientable": round(mean_F_nonor, 4),
            "orientable": round(mean_F_or, 4),
            "n_non_orientable": n_nonor, "n_orientable": n_or,
            "source": attr_rel,
            "locator": "mean of F by label over test_table; label 1 = NON-orientable",
        },
        "triangle_count_distribution": {
            "bins_F": [list(b) for b in BINS],
            "share_orientable": [round(float(v), 4) for v in sh_or],
            "share_non_orientable": [round(float(v), 4) for v in sh_no],
            "source": attr_rel,
            "locator": ("per-class share of test_table complexes with F in "
                        "each bin (label 1 = NON-orientable)"),
        },
        "label_semantics": ("attr-fig JSON: label 1 = majority = NON-orientable; "
                            "p = p(non-orientable); orientable_class field = 0. The "
                            "matched_groups_smallest role names are swapped in the "
                            "frozen JSON and are not used here."),
    },
    "figure_B_mantra_fix": {
        "balanced_accuracy_ladder_3seed_mean": {
            "counts_only": round(bal_count, 4),
            "plus_local_structural": round(bal_struct, 4),
            "plus_signed_hodge_spectral": round(bal_spec, 4),
            "source": led_rel,
            "locator": ("claims ids count_ceiling / orient_struct / orient_spec "
                        "(experiment exp_01kzv8eb90fwat82mt8ke8fj3z; per-seed values "
                        "not in the frozen snapshot, so no error bars are drawn)"),
        },
        "hopse_unsigned_spectral_reference": {
            "balanced_accuracy_mean": round(float(hopse_bal.mean()), 4),
            "sd": round(float(hopse_bal.std(ddof=1)), 4),
            "per_seed": [round(float(v), 4) for v in hopse_bal],
            "source": "data/frozen/hopse_sweep_mantra.parquet",
            "locator": ("per-seed argmax val_accuracy; column test_balanced_accuracy "
                        "(deck value 0.79±0.01)"),
        },
        "chance": 0.5,
    },
}
(OUT / "mantra_numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")

# ================================================================ drawing
SAVE_KW = dict(facecolor=SURFACE, dpi=200)


def rounded_bar(ax, cx, width, y0, y1, color, r_px=5.0, zorder=3):
    """Bar with 4-5px rounded data-end, square at the baseline."""
    if abs(y1 - y0) < 1e-12:
        return
    p2d = ax.transData.transform([[0, 0], [1, 1]])
    rx = r_px / abs(p2d[1, 0] - p2d[0, 0])
    ry = r_px / abs(p2d[1, 1] - p2d[0, 1])
    rx = min(rx, width / 2)
    ry = min(ry, abs(y1 - y0))
    x0, x1 = cx - width / 2, cx + width / 2
    k = 0.5523 # circle-ish bezier
    verts = [(x0, y0), (x1, y0), (x1, y1 - ry),
             (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
             (x0 + rx, y1),
             (x0 + rx - k * rx, y1), (x0, y1 - ry + k * ry), (x0, y1 - ry),
             (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=zorder))


def style_axis(ax, ylim, yticks, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, lw=1.0, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=8)


# ---------------------------------------------------------------- FIGURE A
figA = plt.figure(figsize=(12.8, 6.35))
figA.patch.set_facecolor(SURFACE)
gs = figA.add_gridspec(1, 2, width_ratios=[1.62, 1.0],
                       left=0.062, right=0.978, top=0.715, bottom=0.19,
                       wspace=0.26)
axm = figA.add_subplot(gs[0])
axi = figA.add_subplot(gs[1])

figA.text(0.062, 0.962,
          "Message-passing “skill” on MANTRA orientability is a"
          " dataset-size artifact",
          fontsize=20, fontweight="bold", color=INK, ha="left", va="top")
figA.text(0.062, 0.902,
          "AUROC for detecting non-orientability, TopoBench test split"
          " (10,785 triangulations)",
          fontsize=13.5, color=INK2, ha="left", va="top")

CHANCE = 0.5
style_axis(axm, (0.468, 1.005), np.arange(0.5, 1.01, 0.1), "AUROC")
axm.set_title("Compare within identical (V, E, F) sizes\n"
              "— message-passing skill vanishes",
              fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)

models = ["TopoBench menu\n(selects CWN)", "GCCN", "HOPSE"]
all3 = [float(menu_auroc.mean()), float(gccn_auroc.mean()), float(hopse_auroc.mean())]
sd3 = [float(menu_auroc.std(ddof=1)), float(gccn_auroc.std(ddof=1)),
       float(hopse_auroc.std(ddof=1))]
matched = [cc["menu"], cc["gccn"], cc["hopse"]]

X = np.arange(3) * 1.0
W = 0.30
OFF = 0.175
axm.set_xlim(-0.55, 2.55)
axm.set_xticks(X)
axm.set_xticklabels(models, fontsize=13.5, color=INK)

axm.axhline(CHANCE, color=BASE, lw=1.4, zorder=2)
axm.text(2.53, CHANCE - 0.0135, "chance = 0.50", fontsize=11.5, color=MUTED,
         ha="right", va="top")

figA.canvas.draw()  # finalize layout so pixel radii are correct
for i in range(3):
    rounded_bar(axm, X[i] - OFF, W, CHANCE, all3[i], BLUE)
    rounded_bar(axm, X[i] + OFF, W, CHANCE, matched[i], ORANGE)
    if sd3[i] > 1e-9:
        axm.errorbar(X[i] - OFF, all3[i], yerr=sd3[i], fmt="none",
                     ecolor=INK2, elinewidth=1.6, capsize=5, capthick=1.6,
                     zorder=5)
    top = all3[i] + (sd3[i] if sd3[i] > 1e-9 else 0.0)
    axm.text(X[i] - OFF, top + 0.016, f"{all3[i]:.3f}", fontsize=13.5,
             fontweight="bold", color=INK, ha="center", va="bottom")
    axm.text(X[i] + OFF, max(matched[i], CHANCE) + 0.016, f"{matched[i]:.3f}",
             fontsize=13.5, fontweight="bold", color=INK, ha="center",
             va="bottom")

# tag each collapsed bar right above its value label
axm.text(X[0] + OFF + 0.06, 0.551, "= chance", fontsize=12.5, color=INK2,
         ha="center", va="bottom", fontstyle="italic")
axm.text(X[1] + OFF, 0.555, "≈ chance", fontsize=12.5, color=INK2,
         ha="center", va="bottom", fontstyle="italic")
axm.text(X[2] + OFF + 0.065, 0.848, "real signal\nsurvives", fontsize=12.5,
         color=INK2, ha="center", va="bottom", linespacing=1.25)

legA = axm.legend(handles=[
    mpl.patches.Patch(color=BLUE,
                      label="All test complexes (3-seed mean ± sd)"),
    mpl.patches.Patch(color=ORANGE,
                      label="Within size-matched groups (seed 43)")],
    loc="upper left", frameon=False, fontsize=12.5, handlelength=1.2,
    handleheight=1.0, borderaxespad=0.1, labelcolor=INK2)

# ---- inset: the artifact itself -------------------------------------------
axi.set_facecolor(SURFACE)
axi.set_title("The artifact: non-orientable\ntriangulations are simply larger",
              fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
centers = bin_centers
axi.plot(centers, sh_or, color=GRAY_SERIES, lw=2.2, zorder=3,
         marker="o", ms=7, mec=SURFACE, mew=2)
axi.plot(centers, sh_no, color=VIOLET, lw=2.2, zorder=4,
         marker="o", ms=7, mec=SURFACE, mew=2)
axi.fill_between(centers, 0, sh_or, color=GRAY_SERIES, alpha=0.10, zorder=1)
axi.fill_between(centers, 0, sh_no, color=VIOLET, alpha=0.10, zorder=2)
for s in ("top", "right", "left"):
    axi.spines[s].set_visible(False)
axi.spines["bottom"].set_color(BASE)
axi.grid(axis="y", color=GRID, lw=1.0, zorder=0)
axi.tick_params(axis="y", length=0, labelsize=12)
axi.tick_params(axis="x", length=0, pad=8, labelsize=12.5)
axi.set_ylim(0, 0.76)
axi.set_yticks([0, 0.2, 0.4, 0.6])
axi.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=12)
axi.set_xticks(centers)
axi.set_xticklabels([f"{a}–{b}" for a, b in BINS])
axi.set_xlabel("triangles per complex", fontsize=13.5)
axi.set_ylabel("share of class", fontsize=13.5)

# class labels carry the class means; carets mark them on the axis
axi.text(19.9, 0.542, "orientable\nmean 20.5", fontsize=12.5, color=INK2,
         ha="right", va="bottom", linespacing=1.25)
axi.text(25.4, 0.565, "non-orientable\nmean 23.2", fontsize=12.5, color=INK2,
         ha="center", va="bottom", linespacing=1.25)
axi.plot([mean_F_or], [0.006], marker="^", ms=9, color=GRAY_SERIES,
         clip_on=False, zorder=5)
axi.plot([mean_F_nonor], [0.006], marker="^", ms=9, color=VIOLET,
         clip_on=False, zorder=5)
axi.text(8.3, 0.66, "triangle count alone:\n0.784 AUROC", fontsize=13,
         color=INK, ha="left", va="bottom", fontweight="bold",
         linespacing=1.25)

figA.text(0.062, 0.062,
          "Size-matched = identical (V, E, F): 4 groups, 5,645 complexes,"
          " 1,318,027 cross-class pairs; count-conditional AUROC from the"
          " frozen seed-43 attribution run.",
          fontsize=10.5, color=MUTED, ha="left", va="top")
figA.text(0.062, 0.028,
          "3-seed = seeds 42/43/44, per-seed validation-accuracy selection,"
          " frozen parquets. Test class balance: 9,930 non-orientable /"
          " 855 orientable.",
          fontsize=10.5, color=MUTED, ha="left", va="top")

figA.savefig(OUT / "mantra_finding.png", **SAVE_KW)
figA.savefig(OUT / "mantra_finding.svg", facecolor=SURFACE,
             metadata={"Date": None})
plt.close(figA)

# ---------------------------------------------------------------- FIGURE B
figB = plt.figure(figsize=(12.8, 6.35))
figB.patch.set_facecolor(SURFACE)
axb = figB.add_axes((0.075, 0.185, 0.9, 0.60))

figB.text(0.075, 0.955,
          "Signed Hodge-Laplacian spectra supply the missing global signal",
          fontsize=20, fontweight="bold", color=INK, ha="left", va="top")
figB.text(0.075, 0.885,
          "Balanced accuracy on MANTRA orientability, test split — feature"
          " ladder, 3-seed means",
          fontsize=13.5, color=INK2, ha="left", va="top")

style_axis(axb, (0.468, 1.005), np.arange(0.5, 1.01, 0.1),
           "Balanced accuracy")
stages = ["Cell counts only\n(V, E, F)",
          "+ local structural\nfeatures",
          "+ signed Hodge-Laplacian\neigenvectors"]
vals = [bal_count, bal_struct, bal_spec]
Xb = np.arange(3) * 1.0
axb.set_xlim(-0.62, 2.62)
axb.set_xticks(Xb)
axb.set_xticklabels(stages, fontsize=13.5, color=INK)

axb.axhline(CHANCE, color=BASE, lw=1.4, zorder=2)
axb.text(-0.60, CHANCE - 0.0135, "chance = 0.50", fontsize=11.5, color=MUTED,
         ha="left", va="top")

# HOPSE unsigned-spectral reference band (3-seed mean +/- sd)
h_m = float(hopse_bal.mean())
h_s = float(hopse_bal.std(ddof=1))
axb.axhspan(h_m - h_s, h_m + h_s, color=ORANGE, alpha=0.10, zorder=1)
axb.axhline(h_m, color=ORANGE, lw=1.6, ls=(0, (5, 3)), zorder=2)
axb.text(-0.58, h_m + 0.014,
         "HOPSE — generic unsigned spectral encodings:"
         f" {h_m:.2f} ± {h_s:.2f} (3-seed, sweep val-best)",
         fontsize=12.5, color=INK2, ha="left", va="bottom")

figB.canvas.draw()
WB = 0.42
for i, v in enumerate(vals):
    rounded_bar(axb, Xb[i], WB, CHANCE, v, ORD_RAMP[i])
    axb.text(Xb[i], v + 0.016, f"{v:.3f}", fontsize=15, fontweight="bold",
             color=INK, ha="center", va="bottom")

axb.annotate("", xy=(Xb[2] - WB / 2 - 0.05, bal_spec - 0.012),
             xytext=(Xb[1] + WB / 2 + 0.05, bal_struct + 0.006),
             arrowprops=dict(arrowstyle="->", color=INK2, lw=1.6,
                             connectionstyle="arc3,rad=-0.22"))
axb.text(1.10, 0.690, "signed spectral\nfeatures: +0.34",
         fontsize=13.5, color=INK, ha="center", va="center",
         fontweight="bold", linespacing=1.25)
axb.text(Xb[0], 0.560, "count-only models\n≈ chance", fontsize=12.5, color=INK2,
         ha="center", va="bottom", linespacing=1.25)

figB.text(0.075, 0.062,
          "Ladder bars: 3-seed means from the frozen claims ledger (rescue"
          " experiment); per-seed spread not in the frozen snapshot, so no"
          " error bars are drawn.",
          fontsize=10.5, color=MUTED, ha="left", va="top")
figB.text(0.075, 0.028,
          "HOPSE reference: per-seed validation-accuracy-best, seeds 42/43/44"
          " — unsigned spectral encodings get partway; the signed operator"
          " closes the gap.",
          fontsize=10.5, color=MUTED, ha="left", va="top")

figB.savefig(OUT / "mantra_fix.png", **SAVE_KW)
figB.savefig(OUT / "mantra_fix.svg", facecolor=SURFACE, metadata={"Date": None})
plt.close(figB)

print("wrote", OUT / "mantra_finding.png", "and", OUT / "mantra_fix.png")
