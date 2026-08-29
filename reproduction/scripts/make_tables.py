#!/usr/bin/env python
"""Generate paper tables, result macros, and figure-input frames from
data/frozen/. Every number that appears in the paper flows through here —
nothing is typed into the LaTeX by hand.

    python scripts/make_tables.py --menu-per-seed   # figure input frame
    python scripts/make_tables.py --ceilings        # figure ceilings frame
    python scripts/make_tables.py --macros          # paper/results_macros.tex
    python scripts/make_tables.py --all

Schema notes: the exact key layout inside menu12/analysis12 was recorded in
the handoff at path level (e.g. ``menu12.per_dataset[ds].sets[mask].per_seed``)
but not at leaf level; the leaf-level accessors below are the single place to
reconcile when the real files arrive. Each accessor fails loudly with the
observed layout to make that reconciliation a five-minute job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from toposhap.io import FROZEN_DIR, load_menu12, load_pooled9  # noqa: E402

#: datasets whose honest metric is balanced accuracy (class imbalance);
#: mirror of figures/build_lead_figure.BALANCED_DATASETS.
BALANCED_DATASETS = {"mantra_orientation"}


def _concat_frames(*names: str) -> pd.DataFrame | None:
    """Merge frozen dataset batches — mirror of
    figures/build_lead_figure._concat_frames (same batching, same order)."""
    frames = []
    for n in names:
        f = FROZEN_DIR / n
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _with_score(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add the per-row scored metric: balanced test accuracy on imbalanced
    datasets (rows lacking it are dropped — graceful when a balanced
    backfill has not landed yet), plain test accuracy elsewhere. Mirror of
    figures/build_lead_figure._with_yval."""
    if df is None or df.empty:
        return df
    df = df.copy()
    bal = df.get("test_balanced_accuracy")
    is_bal = df.dataset.isin(BALANCED_DATASETS)
    df["score"] = df.test_accuracy.where(
        ~is_bal, bal if bal is not None else float("nan")
    )
    return df.dropna(subset=["score"])


def _per_seed_leaf(per_seed_node, what: str):
    """Extract {seed: {'val': float, 'test': float}} from a per_seed node,
    tolerating the two layouts seen in this project's files (dict of dicts,
    or parallel lists)."""
    if isinstance(per_seed_node, dict):
        out = {}
        for seed, leaf in per_seed_node.items():
            if isinstance(leaf, dict) and {"val", "test"} <= set(leaf):
                out[int(seed)] = {
                    "val": float(leaf["val"]),
                    "test": float(leaf["test"]),
                }
            else:
                raise KeyError(
                    f"unrecognised per_seed leaf for {what}: {leaf!r} — "
                    "reconcile _per_seed_leaf() with the real schema"
                )
        return out
    raise KeyError(
        f"unrecognised per_seed node type for {what}: {type(per_seed_node)}"
    )


def menu_per_seed() -> Path:
    """Symmetric menu bars: per-seed validation-best selection.

    For each dataset and seed, pick the menu mask with the best *validation*
    score for that seed, then record that mask's *test* score — the honest
    mirror of how the recipe's own bars are computed (per-seed selection, not
    pooled-val-best). True menu cost: training all 6 candidates once, i.e.
    6 x mean sweep-run seconds (FLOPs backfilled when #13's table lands)."""
    from toposhap.io import load_cost_wallclock12

    menu = load_menu12()
    cost = load_cost_wallclock12()["per_dataset"]

    # menu FLOPs: six trainings priced from the #14 per-mask profiles at the
    # mean menu-row epoch count from Stage D (cost13); NaN if either missing
    MENU_MASKS = [21, 26, 72, 129, 329, 479]
    menu_flops_by_ds = {}
    cost13_path = FROZEN_DIR / "hopse13" / "results" / "cost13.parquet"
    if cost13_path.exists():
        c13 = pd.read_parquet(cost13_path)
        menu_rows = c13[c13.approach == "gccn_menu"]
        for ds in menu["per_dataset"]:
            prof_path = (
                FROZEN_DIR / "marker_arm14" / "results" / "flops"
                / f"flops_{ds}.json"
            )
            sub = menu_rows[menu_rows.dataset == ds]
            if not prof_path.exists() or sub.empty:
                continue
            per_mask = json.loads(prof_path.read_text())["per_mask"]
            mean_epochs = float(sub.n_epochs_trained.mean())
            menu_flops_by_ds[ds] = sum(
                per_mask[str(m)]["train_epoch_flops"] * mean_epochs
                for m in MENU_MASKS
                if str(m) in per_mask
            )

    rows = []
    for ds, node in menu["per_dataset"].items():
        wall = 6.0 * cost.get(ds, {}).get("mean_sweep_run_seconds", float("nan"))
        ds_flops = menu_flops_by_ds.get(ds, float("nan"))
        seeds: dict[int, dict[int, dict]] = {}
        for mask, mnode in node["sets"].items():
            for seed, leaf in _per_seed_leaf(
                mnode["per_seed"], f"menu12[{ds}][{mask}]"
            ).items():
                seeds.setdefault(seed, {})[int(mask)] = leaf
        for seed, by_mask in seeds.items():
            best_mask = max(by_mask, key=lambda m: by_mask[m]["val"])
            rows.append(
                dict(
                    dataset=ds,
                    seed=seed,
                    mask=best_mask,
                    test_accuracy=by_mask[best_mask]["test"],
                    flops=ds_flops,
                    wall_seconds=wall,
                )
            )
    out = FROZEN_DIR / "menu_per_seed_selection.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def recipe12() -> Path:
    """Recipe stars from #12 (3 seeds, plain selector) — the pre-#14 figure
    input, so the real-data figure exists before the marker arm lands.

    Star x-position per handoff: stem + 0.1 x game + continuation (the 0.1
    prices the 128-pass sampled game against the measured exact-game clock).
    """
    from toposhap.io import load_analysis12, load_cost_wallclock12

    analysis = load_analysis12()["per_dataset"]
    cost = load_cost_wallclock12()["per_dataset"]
    rows = []
    for ds, node in analysis.items():
        guided = node["method_A"]["cells"]["guided"]
        for k, by_seed in guided.items():
            for seed, leaf in by_seed.items():
                c = cost[ds]["per_seed"][seed]
                wall = (
                    c["stem5_seconds"]
                    + 0.1 * c["game_seconds"]
                    + c["continued_seconds_by_k"][k]
                )
                rows.append(
                    dict(
                        dataset=ds,
                        seed=int(seed),
                        selector="plain",
                        k=int(k),
                        mask=int(leaf["mask"]),
                        val_accuracy=float(leaf["val"]),
                        test_accuracy=float(leaf["test"]),
                        wall_seconds=float(wall),
                        flops=float("nan"),
                    )
                )
    out = FROZEN_DIR / "recipe12.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


#: measured validation-set sizes (denominators of the exact val fractions)
VAL_SIZE = {
    "benzene": 1800, "mutagenicity_gxai": 265, "NCI1": 1027,
    "fluoride_carbonyl": 1300, "cocitation_cora": 677,
    "cocitation_citeseer": 831, "cocitation_pubmed": 4929,
}


def markers14_flops() -> Path:
    """marker_arm14_frame.parquet: the marker-arm frame with assembled FLOPs.

    Per row: flops = stem + game + continuation, where
      stem  = 5 epochs x train_epoch_flops(full mask 2047)
      game  = n_unique_evals x eval_pass_flops   (unique-eval accounting,
              measured for seeds 45/46, budget 128 for #12-era seeds)
      cont  = train_epoch_flops(picked mask) x epochs actually trained
              (measured for 45/46 from continued5; per-(ds,k) mean of those
              as the estimate for 42-44 — x-positions only, flagged in doc).
    Also emits autok_preview.parquet: per-seed auto-k pick (k in {3,6} by the
    leftover-phi < binomial-se rule) for the seeds whose cheap games are in
    the snapshot (45/46), valued at the anchored continuation of that k.
    """
    import numpy as np

    from toposhap.io import load_marker_arm14

    frame = load_marker_arm14()
    root = FROZEN_DIR / "marker_arm14" / "results"
    flops_prof = {}
    for f in (root / "flops").glob("flops_*.json"):
        d = json.loads(f.read_text())
        flops_prof[d["dataset"]] = d["per_mask"]

    # measured continuation epochs (seeds 45/46)
    epochs = {}
    for f in (root / "continued").glob("continued5_*.jsonl"):
        for line in f.read_text().strip().splitlines():
            r = json.loads(line)
            epochs[(r["dataset"], r["seed"], r["mask"])] = r[
                "n_epochs_trained_in_continuation"
            ]

    def epochs_for(ds, seed, mask, k):
        if (ds, seed, mask) in epochs:
            return epochs[(ds, seed, mask)]
        same_k = [
            e for (d2, s2, m2), e in epochs.items()
            if d2 == ds and bin(m2).count("1") == k
        ]
        return float(np.mean(same_k)) if same_k else float("nan")

    m14 = json.loads((root / "markers14.json").read_text())

    def game_evals(ds, seed):
        comp = m14["per_dataset"][ds]["component_seconds"]["sampled_games"]
        if str(seed) in comp:
            g = comp[str(seed)]
            return g["passes_used"] + g.get("extra_nomination_passes", 0)
        return 128  # #12 budget, unique-eval accounting

    rows = []
    for r in frame.to_dict("records"):
        prof = flops_prof[r["dataset"]]
        full = prof["2047"]
        stem_f = 5 * full["train_epoch_flops"]
        game_f = game_evals(r["dataset"], r["seed"]) * full["eval_pass_flops"]
        cont_f = (
            prof[str(r["mask"])]["train_epoch_flops"]
            * epochs_for(r["dataset"], r["seed"], r["mask"], r["k"])
        )
        r["flops"] = stem_f + game_f + cont_f
        rows.append(r)
    out = FROZEN_DIR / "marker_arm14_frame.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")

    # ---- auto-k preview (seeds with cheap games in the snapshot) ----
    preview = []
    for f in (root / "cheap").glob("sampled_*_E5.npz"):
        parts = f.stem.split("_seed")
        ds = parts[0].replace("sampled_", "")
        seed = int(parts[1].split("_")[0])
        z = np.load(f)
        phi = z["est"]
        se = None
        if ds in VAL_SIZE:
            p = float(
                np.mean([r["test_accuracy"] for r in rows
                         if r["dataset"] == ds])
            )
            se = (p * (1 - p) / VAL_SIZE[ds]) ** 0.5
        order = np.argsort(-phi)
        leftover3 = float(np.clip(phi[order[3:]], 0, None).sum())
        k = 3 if (se is not None and leftover3 < se) else 6
        cell = (
            m14["per_dataset"][ds]["per_k_cells"]["anchored"]
            .get(str(k), {})
            .get(str(seed))
        )
        if cell is None:
            continue
        base = [r for r in rows if r["dataset"] == ds and r["seed"] == seed
                and r["selector"] == "anchored" and r["k"] == k]
        preview.append(
            dict(
                dataset=ds, seed=seed, k=k, mask=cell["mask"],
                test_accuracy=cell["test"],
                flops=base[0]["flops"] if base else float("nan"),
                wall_seconds=base[0]["wall_seconds"] if base else float("nan"),
                leftover3=leftover3, binomial_se=se,
            )
        )
    out2 = FROZEN_DIR / "autok_preview.parquet"
    pd.DataFrame(preview).to_parquet(out2, index=False)
    print(f"wrote {out2} ({len(preview)} rows)")
    return out


def autok_marker() -> Path:
    """autok_marker.parquet: the backward-elimination auto-k arm with FLOPs.

    FLOPs per row = 5 x train_epoch_flops(2047) [stem]
                  + n_game_evaluations x eval_pass_flops [elimination walk]
                  + train_epoch_flops(picked mask) x epochs continued.
    Picked masks are mostly unprofiled, so train_epoch_flops(mask) comes from
    a per-dataset linear fit of train_epoch_flops vs coalition size over the
    profiled masks (route GNNs are homogeneous; fit quality printed).
    """
    import numpy as np

    src = FROZEN_DIR / "autok_backward.parquet"
    df = pd.read_parquet(src)
    rows = []
    for ds, grp in df.groupby("dataset"):
        prof = json.loads(
            (FROZEN_DIR / "marker_arm14" / "results" / "flops"
             / f"flops_{ds}.json").read_text()
        )["per_mask"]
        # additive per-route model: flops(mask) ~ intercept + sum_i c_i[bit i]
        masks = np.array([int(m) for m in prof], dtype=int)
        tf = np.array([prof[str(m)]["train_epoch_flops"] for m in masks],
                      dtype=float)
        X = np.column_stack(
            [np.ones(len(masks))]
            + [(masks >> i) & 1 for i in range(11)]
        ).astype(float)
        coef, *_ = np.linalg.lstsq(X, tf, rcond=None)
        pred = X @ coef
        r2 = 1 - ((tf - pred) ** 2).sum() / ((tf - tf.mean()) ** 2).sum()

        def fit_flops(mask: int) -> float:
            x = np.array([1.0] + [(mask >> i) & 1 for i in range(11)])
            return float(x @ coef)

        full = prof["2047"]
        for r in grp.to_dict("records"):
            train_f = prof.get(str(r["mask"]), {}).get(
                "train_epoch_flops", fit_flops(int(r["mask"]))
            )
            r["flops"] = (
                5 * full["train_epoch_flops"]
                + r["n_game_evaluations"] * full["eval_pass_flops"]
                + train_f * r["n_epochs_trained_in_continuation"]
            )
            rows.append(r)
        print(f"  {ds}: flops-vs-k fit R^2 = {r2:.4f}")
    out = FROZEN_DIR / "autok_marker.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def ladder_b2_flops() -> Path:
    """Backfill EXACT FLOPs into ladder_b2_marker.parquet (seeds 42-44 only).

    flops = 5 x train_epoch_flops(2047) + n_evals x eval_pass_flops
            + BOTH rungs' train_epoch_flops(mask) x measured epochs.
    All per-mask values come from measured profiles: results/flops_local/
    (this-cluster FlopCounterMode runs over every rung mask, seeds 42-44)
    with the #14 profiles as a secondary source. NO fitted values: a rung
    mask absent from both profile sets is an error, not an estimate.
    Researcher directive 2026-08-17: n=3 seeds {42,43,44} on every marker.
    """
    df = pd.read_parquet(FROZEN_DIR / "ladder_b2_marker.parquet")
    df = df[df.seed.isin([42, 43, 44])].copy()
    jdir = REPO / "results" / "ladder_b2"
    cells = {}
    for f in jdir.glob("ladder_b2_*.jsonl"):
        r = json.loads(f.read_text().strip().splitlines()[-1])
        cells[(r["dataset"], int(r["seed"]))] = r

    def epochs_of(rung: dict, cell: dict, tag: str) -> float:
        for key in (f"n_epochs_trained_in_continuation_{tag}",
                    f"n_epochs_{tag}"):
            if key in cell:
                return cell[key]
        return rung.get("n_epochs_trained_in_continuation",
                        rung.get("n_epochs", float("nan")))

    profs: dict[str, dict] = {}

    def train_flops(ds: str, mask: int) -> float:
        if ds not in profs:
            merged = {}
            p14 = (FROZEN_DIR / "marker_arm14" / "results" / "flops"
                   / f"flops_{ds}.json")
            if p14.exists():
                merged.update(json.loads(p14.read_text())["per_mask"])
            ploc = REPO / "results" / "flops_local" / f"flops_local_{ds}.json"
            if ploc.exists():
                merged.update(json.loads(ploc.read_text())["per_mask"])
            profs[ds] = merged
        entry = profs[ds].get(str(mask))
        if entry is None:
            raise KeyError(
                f"{ds}: mask {mask} unprofiled — extend the flops_local "
                "profiling pass; fitted values are not permitted"
            )
        return entry["train_epoch_flops"]

    rows = []
    for r in df.to_dict("records"):
        ds = r["dataset"]
        cell = cells.get((ds, int(r["seed"])), {})
        full_train = train_flops(ds, 2047)
        eval_pass = profs[ds]["2047"]["eval_pass_flops"]
        n_evals = cell.get("n_game_evaluations", 66)
        total = 5 * full_train + n_evals * eval_pass
        for tag in ("rung1", "rung2"):
            rung = cell.get(tag, {})
            if rung:
                total += train_flops(ds, int(rung["mask"])) * epochs_of(
                    rung, cell, tag
                )
        r["flops"] = total
        rows.append(r)
    out = FROZEN_DIR / "ladder_b2_marker.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows, exact flops, seeds 42-44)")
    return out


def ceilings() -> Path:
    """Dotted dark = exact 6-space optimum (from pooled.json); light gray =
    best-known 11-space lower bound (from #14 landscape when ingested)."""
    pooled = load_pooled9()
    rows = []
    landscape11 = FROZEN_DIR / "landscape11.json"
    best11_by_ds = {}
    if landscape11.exists():
        data = json.loads(landscape11.read_text())
        for ds, node in data.items():
            best11_by_ds[ds] = max(float(v) for v in node["values"].values())
    for ds, node in pooled["retraining_pooled"].items():
        vm = node["values_mean"]
        exact6 = max(float(v) for v in (vm if isinstance(vm, list) else vm.values()))
        rows.append(
            dict(dataset=ds, exact6=exact6, best11=best11_by_ds.get(ds))
        )
    out = FROZEN_DIR / "ceilings.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def _derived_hopse_best() -> pd.DataFrame | None:
    """hopse_best baseline rows for the campaign datasets, derived from the
    frozen HOPSE sweep batches exactly as figures/build_lead_figure does:
    per-seed validation-best (plain val, as wired — the balanced metric
    never re-selects) at the true run-the-whole-grid-once FLOPs."""
    extra = []
    for pq in ("hopse_sweep_t1.parquet", "hopse_sweep_mantra.parquet",
           "hopse_sweep_citations.parquet"):
        sw = _concat_frames(pq)
        if sw is None:
            continue
        key = "subset_identity" if "subset_identity" in sw.columns else "mask"
        for ds, sub in sw.groupby("dataset"):
            grid_flops = float(sub.groupby(key).flops.mean().sum())
            for seed, g in sub.groupby("seed"):
                best = g.loc[g.val_accuracy.idxmax()]
                extra.append(dict(
                    dataset=ds, baseline="hopse_best",
                    n_configs=sub[key].nunique(), flops=grid_flops,
                    wall_seconds=float("nan"), flops_estimated=False,
                    seed=int(seed),
                    test_accuracy=float(best.test_accuracy),
                    val_accuracy=float(best.val_accuracy),
                    test_balanced_accuracy=float(
                        best.get("test_balanced_accuracy", float("nan"))),
                    winner=str(best.get(key, ""))))
    return pd.DataFrame(extra) if extra else None


PENDING_COSTS: list[str] = []


def performance_table() -> Path:
    """Emit paper/tables/performance_summary.tex (Table 2 of the paper) over
    all 11 campaign datasets, and refresh the headline macros it feeds.

    Rows = datasets; columns = ours (ladder B=1 from autok_marker, B=2 from
    the merged ladder_b2 batches, plus the HOPSE fixed-k=3-rung arm) and the
    three baseline sweeps (baseline_markers + hopse_best derived from the
    frozen HOPSE sweep batches), each as mean +- sd test accuracy over seeds
    with FLOPs as a multiple of the B=1 ladder run (B=2 ladder run, marked,
    on the four campaign datasets where B=1 was not run). mantra_orientation
    is scored in BALANCED test accuracy on every entry (plain accuracy is
    majority-collapsed there; per-seed record at
    results/mantra_prepare_logs/class_balance.json); entries whose balanced
    backfill has not landed are dropped to --- rather than mis-scored.
    Values wrapped in \\unfrozen{} until the freeze flips
    \\FrozenResultstrue."""
    display = {
        "benzene": "Benzene",
        "fluoride_carbonyl": "Fluoride-Carbonyl",
        "mutagenicity_gxai": "Mutagenicity",
        "NCI1": "NCI1",
        "MUTAG": "MUTAG",
        "NCI109": "NCI109",
        "alkane_carbonyl": "Alkane-Carbonyl",
        "cocitation_cora": "Cora",
        "cocitation_citeseer": "Citeseer",
        "cocitation_pubmed": "Pubmed",
        "mantra_orientation": "MANTRA",
    }
    order = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai",
             "alkane_carbonyl",
             "NCI1", "MUTAG", "NCI109",
             "cocitation_cora", "cocitation_citeseer",
             "cocitation_pubmed",
             "mantra_orientation"]
    groups = {"benzene": "GraphXAI benchmarks (lifted to cellular "
                         "complexes)",
              "NCI1": "TopoBench benchmarks (lifted to cellular "
                      "complexes)",
              "mantra_orientation": "Natively higher-order (simplicial)"}
    # campaign datasets without a measured B=1 arm get a dagger: cost
    # multiples there are relative to the B=2 ladder run.

    b1 = pd.read_parquet(FROZEN_DIR / "autok_marker.parquet")
    if "selector" in b1.columns:
        b1 = b1[b1.selector == "autok_backward"]
    # merged batches, exactly as figures/build_lead_figure._concat_frames
    b2 = _concat_frames("ladder_b2_marker.parquet",
                        "ladder_b2_t1.parquet",
                        "ladder_b2_mantra.parquet")
    k3 = _concat_frames("hopse_k3_rung.parquet")
    hb2 = _concat_frames("ladder_b2_hopse.parquet",
                         "ladder_b2_hopse_t1.parquet",
                         "ladder_b2_hopse_mantra.parquet")
    bm = pd.read_parquet(FROZEN_DIR / "baseline_markers.parquet")
    bm = bm[bm.baseline != "originals_best"]
    extra = _derived_hopse_best()
    if extra is not None:
        bm = pd.concat([bm, extra], ignore_index=True)
    # GCCN sweep cells for the enumerated datasets: best of the frozen
    # 64-config retraining enumeration (pooled.json); cost estimated at
    # 64 runs (per-run FLOPs scaled from NCI1's measured sweep by graph
    # count) and marked ^e in the table.
    import json as _json
    _pooled = _json.loads((FROZEN_DIR / "pooled.json").read_text())
    _retr = _pooled["retraining_pooled"]
    _nci1 = bm[(bm.dataset == "NCI1") & (bm.baseline == "gccn_sweep_best")]
    _percfg = float(_nci1.flops.iloc[0]) / float(_nci1.n_configs.iloc[0])
    _ngraphs = {"alkane_carbonyl": 1125, "MUTAG": 188, "NCI109": 4127}
    _enum_rows = []
    for _ds, _n in _ngraphs.items():
        _d = _retr[_ds]
        _best = max(_d["values_mean"])
        _sd = float(_d["tolerance"]["seed_sd_median_over_coalitions"])
        _fl = 64 * _percfg * _n / 4110
        for _y in (_best - _sd, _best, _best + _sd):
            _enum_rows.append(dict(dataset=_ds,
                                   baseline="gccn_sweep_best",
                                   n_configs=64, flops=_fl,
                                   wall_seconds=None,
                                   flops_estimated=True, seed=-1,
                                   test_accuracy=_y, val_accuracy=None,
                                   winner="enum_best"))
    bm = pd.concat([bm, pd.DataFrame(_enum_rows)], ignore_index=True)

    b1, b2, k3, hb2, bm = (_with_score(f)
                           for f in (b1, b2, k3, hb2, bm))

    def agg(df):
        return df.groupby("dataset").agg(
            acc=("score", "mean"),
            sd=("score", "std"),
            flops=("flops", "mean"),
        )

    g1, g2, g3, gh = agg(b1), agg(b2), agg(k3), agg(hb2)
    gb = bm.groupby(["dataset", "baseline"]).agg(
        acc=("score", "mean"),
        sd=("score", "std"),
        flops=("flops", "mean"),
    )

    def cell(acc, sd, mult):
        return (
            f"\\unfrozen{{{acc:.3f}}}$\\pm${sd:.3f} "
            f"{{\\scriptsize($\\times${mult:.1f})}}"
        )

    lines = [
        "% " + "=" * 70,
        "% tables/performance_summary.tex",
        "% *** AUTO-GENERATED by scripts/make_tables.py"
        " --performance-table ***",
        "% *** from data/frozen/{autok_marker,ladder_b2_{marker,t1,mantra},"
        "hopse_k3_rung, ***",
        "% *** baseline_markers,hopse_sweep_{t1,mantra}}."
        " DO NOT EDIT BY HAND. ***",
        "% " + "=" * 70,
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Architecture selection.} Test accuracy",
        "  (mean $\\pm$ sd over seeds \\unfrozen{\\ResSeedList}) and, in",
        "  parentheses, FLOPs as a multiple of the $B{=}1$ selection run",
        "  (\\cref{app:flops-conventions}). Bold marks the best",
        "  accuracy in each row; gray marks entries within one sd of",
        "  it. The last column quantifies the tradeoff: the accuracy",
        "  of the best selection arm minus the best sweep's, and, in",
        "  parentheses, that sweep's compute as a multiple of that",
        "  selection arm's. Sweeps are priced at their full grid cost.",
        "  --- = arm not defined or not run;",
        "  $^{\\dagger}$no $B{=}1$ arm, multiples relative to $B{=}2$;",
        "  $^{e}$best of the 64-config retraining enumeration at",
        "  estimated cost; $^{b}$balanced test accuracy (plain accuracy",
        "  collapses to the majority class).}",
        "\\label{tab:performance-summary}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Ours (selection, GCCN)}"
        " & \\multicolumn{1}{c}{Ours (selection, HOPSE)}"
        " & \\multicolumn{2}{c}{Sweeps (at full grid cost)}"
        " & \\multicolumn{1}{c}{Tradeoff} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}"
        "\\cmidrule(lr){7-7}",
        "Dataset & $B{=}1$ & $B{=}2$ & $B{=}2$ & GCCN sweep"
        " & HOPSE sweep & $\\Delta$ (sweep cost) \\\\",
        "\\midrule",
    ]
    enum_ds = {"alkane_carbonyl", "MUTAG", "NCI109"}
    row_gaps = []  # (best sweep - best selection, sweep sd) per dataset row
    for i, ds in enumerate(order):
        if ds in groups:
            if i:
                lines.append("\\midrule")
            lines.append("\\multicolumn{7}{l}{\\emph{" + groups[ds]
                         + "}} \\\\")
        has_b1 = ds in g1.index
        r2 = g2.loc[ds]
        base = g1.loc[ds].flops if has_b1 else r2.flops
        marks = (["b"] if ds in BALANCED_DATASETS else []) + (
            [] if has_b1 else ["\\dagger"])
        name = display[ds] + (
            "$^{" + ",".join(marks) + "}$" if marks else "")
        vals = [
            (g1.loc[ds].acc, g1.loc[ds].sd, 1.0, "") if has_b1 else None,
            (r2.acc, r2.sd, r2.flops / base, ""),
            ((gh.loc[ds].acc, gh.loc[ds].sd, gh.loc[ds].flops / base, "")
             if ds in gh.index else None),
        ]
        for bl in ("gccn_sweep_best", "hopse_best"):
            if (ds, bl) in gb.index:
                rb = gb.loc[(ds, bl)]
                suf = ("$^{e}$" if bl == "gccn_sweep_best"
                       and ds in enum_ds else "")
                vals.append((rb.acc, rb.sd, rb.flops / base, suf))
            else:
                vals.append(None)
        present = [v for v in vals if v is not None]
        best_acc, best_sd = max((v[0], v[1]) for v in present)
        row = [name]
        for i, v in enumerate(vals):
            if v is None:
                row.append("---")
                continue
            acc, sd, mult, suf = v
            body = f"\\unfrozen{{{acc:.3f}}}$\\pm${sd:.3f}"
            if acc == best_acc:
                body = f"\\textbf{{{body}}}"
            if pd.notna(mult):
                body += f" {{\\scriptsize($\\times${mult:.1f})}}"
            else:
                PENDING_COSTS.append(f"{ds}: cost multiple missing")
            body += suf
            if acc >= best_acc - best_sd:
                body = "\\cellcolor{black!10}" + body
            row.append(body)
        # tradeoff column: best selection arm vs best sweep arm, and the
        # sweep's compute multiple relative to that selection arm
        ours = [v for v in vals[:3] if v is not None and pd.notna(v[2])]
        sweeps = [v for v in vals[3:] if v is not None and pd.notna(v[2])]
        if ours and sweeps:
            o = max(ours, key=lambda v: v[0])
            s = max(sweeps, key=lambda v: v[0])
            d = o[0] - s[0]
            row_gaps.append((s[0] - o[0], s[1]))
            row.append(f"\\unfrozen{{{d:+.3f}}} "
                       f"{{\\scriptsize($\\times${s[2] / o[2]:.0f})}}")
        else:
            row.append("---")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    out = REPO / "paper" / "tables" / "performance_summary.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    if PENDING_COSTS:
        print("WARNING missing cost multiples (parenthetical omitted):",
              file=sys.stderr)
        for m in PENDING_COSTS:
            print(f"  {m}", file=sys.stderr)

    _refresh_headline_macros(g1, g2, g3, gb, b2, k3, gh,
                             row_gaps)
    return out


def _refresh_headline_macros(g1, g2, g3, gb, b2, k3, gh=None,
                             row_gaps=()) -> None:
    """Recompute the Table-2 headline aggregates and the campaign-11 claim
    macros in paper/results_macros.tex from the same merged frames the table
    is built from (still UNFROZEN until scripts/freeze_results.py runs)."""
    import numpy as np

    gs = gb.xs("gccn_sweep_best", level="baseline")
    both = [ds for ds in gs.index if ds in g1.index]
    sweep_mult = {ds: gs.loc[ds].flops / g1.loc[ds].flops for ds in both}
    b2_mult = {ds: g2.loc[ds].flops / g1.loc[ds].flops for ds in g1.index}
    sweep_b2 = {ds: gs.loc[ds].flops / g2.loc[ds].flops for ds in gs.index
                if ds in g2.index}

    updates = {
        # ---- Table 2 headline aggregates (see the macro-block comment) ----
        "ResSweepCostMultMin": f"{np.floor(min(sweep_mult.values())):.0f}",
        "ResSweepCostMultMax": f"{np.ceil(max(sweep_mult.values())):.0f}",
        "ResLadderBTwoCostMult": f"{np.median(list(b2_mult.values())):.1f}",
        "ResSweepVsBTwoMin": f"{min(sweep_b2.values()):.1f}",
        "ResSweepVsBTwoMax": f"{max(sweep_b2.values()):.1f}",
        "ResCoraGap": f"{gs.loc['cocitation_cora'].acc - g2.loc['cocitation_cora'].acc:.3f}",
        "ResCiteseerGap": f"{gs.loc['cocitation_citeseer'].acc - g2.loc['cocitation_citeseer'].acc:.3f}",
        "ResFluorideBOneSd": f"{g1.loc['fluoride_carbonyl'].sd:.3f}",
        "ResFluorideBTwoAcc": f"{g2.loc['fluoride_carbonyl'].acc:.3f}",
        "ResFluorideBTwoSd": f"{g2.loc['fluoride_carbonyl'].sd:.3f}",
        # HOPSE-family ladder now runs on the 6 molecule datasets + MANTRA
        "ResMoleculeDatasets": str(
            k3.dataset.nunique() - int("mantra_orientation" in set(k3.dataset))
        ),
        "ResLeadPanels": str(g2.index.nunique()),
    }

    # ---- selection-vs-sweep honesty macros (per dataset row) --------------
    # gap = best sweep mean - best selection mean, exactly as in the table's
    # tradeoff column; a row is "within" when the selection matches or
    # exceeds the best sweep within one sweep seed-sd. Feeds section 6.
    within = sum(1 for gap, sd in row_gaps if gap <= sd)
    updates["ResSelMaxSweepGap"] = f"{max(g for g, _ in row_gaps):.3f}"
    updates["ResSelWithinRows"] = str(within)
    updates["ResSelTotalRows"] = str(len(row_gaps))

    # ---- MUTAG: the off-suite fixed-k tail, measured live ----------------
    anch = _with_score(_concat_frames("anchored3_exact.parquet",
                                      "anchored3_exact_t1.parquet",
                                      "anchored3_exact_mantra.parquet"))
    mt = anch[anch.dataset == "MUTAG"].score
    updates["ResMutagFixedKLiveAcc"] = f"{mt.mean():.3f}"
    updates["ResMutagFixedKLiveSd"] = f"{mt.std(ddof=1):.3f}"
    mb = b2[b2.dataset == "MUTAG"].score
    updates["ResMutagLadderBTwoAcc"] = f"{mb.mean():.3f}"
    updates["ResMutagLadderBTwoSd"] = f"{mb.std(ddof=1):.3f}"
    ceil = pd.read_parquet(FROZEN_DIR / "ceilings.parquet")
    updates["ResMutagExactOpt"] = (
        f"{float(ceil[ceil.dataset == 'MUTAG'].exact6.iloc[0]):.3f}"
    )

    # ---- MANTRA: the measured leg-2 cross-family gap (balanced acc) ------
    def _bal(df):
        s = df[df.dataset == "mantra_orientation"].score
        return (f"{s.mean():.3f}", f"{s.std(ddof=1):.3f}") if len(s) else None

    for name, df in (("GCCNLadder", b2), ("HopseKRung", k3), ("Anchored", anch)):
        v = _bal(df)
        if v:
            updates[f"ResMantra{name}Bal"] = v[0]
            updates[f"ResMantra{name}BalSd"] = v[1]
    oh = _with_score(_concat_frames("ladder_b2_hopse.parquet",
                                    "ladder_b2_hopse_t1.parquet",
                                    "ladder_b2_hopse_mantra.parquet"))
    v = _bal(oh)
    if v:
        updates["ResMantraHopseLadderBal"] = v[0]
        updates["ResMantraHopseLadderBalSd"] = v[1]
    if ("mantra_orientation", "hopse_best") in gb.index:
        rb = gb.loc[("mantra_orientation", "hopse_best")]
        updates["ResMantraHopseSweepBal"] = f"{rb.acc:.3f}"
        updates["ResMantraHopseSweepBalSd"] = f"{rb.sd:.3f}"
        hl = oh[oh.dataset == "mantra_orientation"]
        if not hl.empty:
            updates["ResMantraSweepVsLadderMult"] = (
                f"{rb.flops / hl.flops.mean():.1f}"
            )

    # majority rate (the collapsed-classifier reference) from the frozen
    # class-balance record; collapsed-model count = frozen cells whose plain
    # test accuracy is byte-identical to it (menu parquet + ladder rungs)
    bal_rec = REPO / "results" / "mantra_prepare_logs" / "class_balance.json"
    if bal_rec.exists():
        rec = json.loads(bal_rec.read_text())
        maj = float(rec["seed42"]["test"]["majority_rate"])
        updates["ResMantraMajorityRate"] = f"{maj:.3f}"
        n_collapsed = 0
        menu = FROZEN_DIR / "gccn_menu_mantra.parquet"
        if menu.exists():
            m = pd.read_parquet(menu)
            n_collapsed += int((m.test_accuracy - maj).abs().lt(1e-6).sum())
        for f in (REPO / "results" / "ladder_b2").glob(
                "ladder_b2_mantra_orientation_seed*.jsonl"):
            r = json.loads(f.read_text().strip().splitlines()[-1])
            for tag in ("rung1", "rung2"):
                if abs(r.get(tag, {}).get("test", 0.0) - maj) < 1e-6:
                    n_collapsed += 1
        updates["ResMantraCollapsedModels"] = str(n_collapsed)

    _patch_macros(updates)


def _patch_macros(updates: dict[str, str]) -> None:
    """Rewrite \\newcommand values in paper/results_macros.tex in place,
    preserving trailing comments; unknown names must already have a
    definition line (added with its documentation comment by hand once)."""
    import re

    tex = REPO / "paper" / "results_macros.tex"
    text = tex.read_text()
    missing = []
    for name, value in updates.items():
        pat = re.compile(
            r"(\\newcommand\{\\" + name + r"\})\{[^{}]*\}")
        if not pat.search(text):
            missing.append(name)
            continue
        text = pat.sub(lambda m: m.group(1) + "{" + value + "}", text)
    tex.write_text(text)
    print(f"patched {len(updates) - len(missing)} macros in {tex}")
    if missing:
        raise SystemExit(
            f"macros with no definition line in results_macros.tex: "
            f"{missing} — add them (with comments) before regenerating"
        )


def macros() -> Path:
    """Regenerate paper/results_macros.tex from frozen data.

    Only numbers whose source files are ingested get (re)computed; the rest
    keep their handoff-sourced UNFROZEN defaults. The manifest hash is stamped
    so the paper records exactly which freeze it was built from."""
    manifest_path = FROZEN_DIR / "MANIFEST.json"
    stamp = "UNFROZEN — no manifest"
    if manifest_path.exists():
        stamp = json.loads(manifest_path.read_text())["manifest_sha256"][:16]

    tex = REPO / "paper" / "results_macros.tex"
    if not tex.exists():
        print(f"{tex} not found (paper skeleton not built yet); skipping")
        return tex
    lines = tex.read_text().splitlines()
    out = []
    replaced = False
    for line in lines:
        if line.startswith("\\newcommand{\\ResManifestHash}"):
            out.append(
                "\\newcommand{\\ResManifestHash}{" + stamp + "}"
            )
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append("\\newcommand{\\ResManifestHash}{" + stamp + "}")
    tex.write_text("\n".join(out) + "\n")
    print(f"stamped manifest {stamp} into {tex}")
    # TODO(freeze): recompute every \Res* macro from frozen frames here and
    # flip \FrozenResultstrue once scripts/freeze_results.py has run.
    return tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--menu-per-seed", action="store_true")
    ap.add_argument("--recipe12", action="store_true")
    ap.add_argument("--markers14", action="store_true")
    ap.add_argument("--autok", action="store_true")
    ap.add_argument("--ladder-b2", action="store_true")
    ap.add_argument("--ceilings", action="store_true")
    ap.add_argument("--performance-table", action="store_true")
    ap.add_argument("--macros", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ran = False
    if args.menu_per_seed or args.all:
        menu_per_seed()
        ran = True
    if args.recipe12 or args.all:
        recipe12()
        ran = True
    if args.markers14 or args.all:
        markers14_flops()
        ran = True
    if args.autok or args.all:
        autok_marker()
        ran = True
    if args.ladder_b2 or args.all:
        ladder_b2_flops()
        ran = True
    if args.ceilings or args.all:
        ceilings()
        ran = True
    if args.performance_table or args.all:
        performance_table()
        ran = True
    if args.macros or args.all:
        macros()
        ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
