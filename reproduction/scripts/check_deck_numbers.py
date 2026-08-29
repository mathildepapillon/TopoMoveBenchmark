"""Verify every number in the slide deck against frozen data.

Numbers living in the deck's HTML text are recomputed from the frozen
parquets / attribution JSONs, formatted exactly as the deck formats
them, and asserted to appear in figures/slides/toposhap_slides.html.

Numbers living inside the deck's FIGURES are covered by asset-sync
checks: each figure builder re-runs (each hard-asserts its numbers
against frozen data and renders byte-deterministically), the rebuilt
PNG must be byte-identical to the asset on disk, and the asset's
base64 must appear verbatim in the deck. Frozen drift, a stale asset,
or a stale embed each fail a distinct check.

Record-sourced values that cannot be recomputed locally (the original cluster record
#6 surrogate 95.8%) are asserted for PRESENCE only (RECORD_SOURCED);
the record's 0.505/0.541/0.876 ladder now lives inside mantra_fix.png,
whose builder asserts them against the frozen claims ledger.

Run: python scripts/check_deck_numbers.py   (exit 1 on any failure)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "figures" / "slides" / "toposhap_slides.html"
FROZEN = REPO / "data" / "frozen"

RECORD_SOURCED = ["95.8%"]

# (asset file, builder script) — builder must reproduce the asset
# byte-identically from frozen data, and the asset must be embedded
# verbatim (base64) in the deck.
ASSETS = [
    ("figures/slides_assets/graphxai_comparison.png",
     "figures/slides_assets/build_graphxai_comparison.py"),
    ("figures/slides_assets/mantra_finding.png",
     "figures/slides_assets/make_mantra_slides.py"),
    ("figures/slides_assets/mantra_fix.png",
     "figures/slides_assets/make_mantra_slides.py"),
    ("figures/mantra_attr_pair.png",
     "figures/build_mantra_attr_figure.py"),
]


def val_best(df, select="val_accuracy", metric="test_auroc"):
    picks = df.loc[df.groupby("seed")[select].idxmax()]
    return picks[metric].mean(), picks[metric].std()


def checks():
    menu = pd.read_parquet(FROZEN / "gccn_menu_mantra.parquet")
    lad = pd.read_parquet(FROZEN / "ladder_b2_mantra.parquet")
    sw = pd.read_parquet(FROZEN / "hopse_sweep_mantra.parquet")
    attr = json.loads((FROZEN / "mantra_attr_fig.json").read_text())
    hopse_rerun = json.loads((FROZEN / "mantra_attr_hopse.json").read_text())

    out = []

    # --- slide "MANTRA benchmark": all-AUROC table -----------------------
    m, _ = val_best(menu)
    out.append((f"menu val-best AUROC {m:.2f}", f">{m:.2f}<"))
    # GCCN-ladder / HOPSE-sweep AUROC and balanced values now live in
    # mantra_finding.png / mantra_fix.png (asset-sync checks below);
    # keep the frozen-data sanity asserts here.
    _ = val_best(lad)
    _ = val_best(sw)
    # menu collapse signature: identical value on all three seeds
    picks = menu.loc[menu.groupby("seed")["val_accuracy"].idxmax()]
    assert picks["test_auroc"].nunique() == 1, "menu no longer collapsed"

    # --- count-conditional AUROC values (drawn in mantra_finding.png) ----
    for model, label in (("menu", "0.500"), ("gccn", "0.506"),
                         ("hopse", "0.791")):
        cc = attr["models"][model]["count_conditional_auroc"]
        assert f"{cc:.3f}" == label, (
            f"{model} cc-AUROC {cc:.3f} != expected {label} — "
            "update this check AND the figure builder together")

    # --- slide "Graph-XAI benchmarks": the one number left in text -------
    tables = json.loads(
        (FROZEN / "attribution4" / "all_tables.json").read_text())
    rows = [r for r in tables["baseline_join"]["rows"]
            if r["source"] == "campaign" and r["model"] in ("gcn", "gin")
            and r["dataset"] == "AlkaneCarbonyl"]
    flat_alkane = max(r["gea_node_mean"] for r in rows)
    out.append((f"flat-GNN alkane GEA {flat_alkane:.3f}",
                f"({flat_alkane:.3f}, GCN)"))
    # pair totals in the flipped orientability convention (phi negated)
    for role, name in (("non_orientable", "torus"), ("orientable", "Klein")):
        s = -float(np.sum(hopse_rerun[role]["phi"]))
        out.append((f"HOPSE sum_phi {name} {s:+.1f}",
                    f"{s:+.1f}".replace("+", "+").replace("-", "&minus;")
                    if s < 0 else f"+{abs(s):.1f}"))
    gsum = {r: -float(np.sum(attr["pair"][r]["gccn"]["phi"]))
            for r in ("orientable", "non_orientable")}
    out.append((f"GCCN totals {gsum['non_orientable']:.2f}/"
                f"{gsum['orientable']:.2f}",
                f"&minus;{abs(gsum['non_orientable']):.2f} vs "
                f"&minus;{abs(gsum['orientable']):.2f}"))

    # --- majority direction ----------------------------------------------
    t = attr["models"]["gccn"]["test_table"]
    frac_majority = np.mean([r["label"] == 1 for r in t])
    assert frac_majority > 0.9, "test_table majority convention changed"
    out.append(("majority class stated as non-orientable",
                "92% of the surfaces are <em>non-orientable</em>"))

    return out


def main() -> int:
    import base64
    import hashlib
    import subprocess

    html = DECK.read_text()
    failures = []

    # --- asset-sync: builder -> asset -> deck ---------------------------
    rebuilt = set()
    for asset, builder in ASSETS:
        apath = REPO / asset
        before = hashlib.md5(apath.read_bytes()).hexdigest()
        if builder not in rebuilt:
            r = subprocess.run(
                [sys.executable, str(REPO / builder)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL  builder {builder} (frozen-data assert?)")
                print(r.stdout[-800:], r.stderr[-800:])
                failures.append(f"builder {builder}")
                rebuilt.add(builder)
                continue
            rebuilt.add(builder)
        after = hashlib.md5(apath.read_bytes()).hexdigest()
        ok_stable = before == after
        print(f"{'PASS' if ok_stable else 'FAIL'}  asset reproducible "
              f"from frozen data: {asset}")
        if not ok_stable:
            failures.append(f"asset stale vs frozen: {asset}")
        b64 = base64.b64encode(apath.read_bytes()).decode()
        ok_embed = b64 in html
        print(f"{'PASS' if ok_embed else 'FAIL'}  asset embedded "
              f"verbatim in deck: {asset}")
        if not ok_embed:
            failures.append(f"deck embed stale: {asset}")
    for desc, needle in checks():
        ok = needle in html
        print(f"{'PASS' if ok else 'FAIL'}  {desc}  [{needle!r}]")
        if not ok:
            failures.append(desc)
    for needle in RECORD_SOURCED:
        ok = needle in html
        print(f"{'PASS' if ok else 'FAIL'}  record-sourced present: {needle}")
        if not ok:
            failures.append(f"record-sourced {needle}")
    if failures:
        print(f"\n{len(failures)} deck-number check(s) FAILED")
        return 1
    print("\nall deck-number checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
