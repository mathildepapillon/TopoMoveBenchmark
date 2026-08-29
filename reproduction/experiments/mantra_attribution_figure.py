"""Per-cell TopoSHAP attributions + counting evidence on MANTRA (deck figure).

Explains three seed-43 models, chosen exactly as the campaign's as-wired
plain-val selection chose them (no cherry-picking beyond fixing the median
seed, disclosed on the slide):

  menu  -- GCCN backbone pruned to the seed-43 val-best menu mask
           (a published architecture; the collapsed counter)
  gccn  -- ladder-selected rung (seed 43: rung1, its frozen mask)
  hopse -- seed-43 val-best sweep configuration

For a matched pair of test complexes with IDENTICAL (V, E, F) cell counts
and opposite orientability labels, computes per-cell Shapley attributions
under the original leg-1 semantics (encoded-feature masking, complex-mean
baseline; ``CellMaskingGame`` parity-proven vs the record's explainer).
The explained scalar is the orientability logit margin
logit[orientable] - logit[non-orientable] (presentation figure, disclosed;
NOT a frozen-number pipeline).

Counting evidence over the full test split, per model:
  * p(orientable) for every test complex + its (V, E, F) counts;
  * count-conditional AUROC: over groups of test complexes with identical
    (V, E, F) containing both classes, the probability that an orientable
    complex outscores a non-orientable one. A model that only counts
    cannot beat 0.5 here by construction.
  * count-surrogate agreement (local recompute of the record-#3 probe):
    logistic model on (V, E, F) fit to the model's TRAIN-split
    predictions, agreement measured on its TEST-split predictions.

Writes results/mantra_attr_fig/attr_fig_data.json. Read-only w.r.t. all
campaign artifacts; safe to run alongside the AUROC backfill.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import os

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import (  # noqa: E402
    DS,
    R,
    SCRATCH,
    find_ckpt,
    gccn_compose,
    gccn_ladder_paths,
    load_ckpt,
    menu_paths,
    sweep_paths,
)
from run_hopse_sweep import compose as sweep_compose  # noqa: E402
from run_hopse_sweep import sweep_configs  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.explain import CellMaskingGame  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402

SEED = 43  # median campaign seed (disclosed on the slide)
PASSES = 512  # per-cell estimator passes; tiny complexes, so still cheap
OUT = R / "mantra_attr_fig"
RANKS = (0, 1, 2)


# ---------------------------------------------------------------- loaders --
def single_batch(loader, idx: int):
    """Collate dataset item ``idx`` as a batch of one."""
    return loader.collate_fn([loader.dataset[idx]])


def batch_counts(batch) -> tuple[int, ...]:
    return tuple(int(getattr(batch, f"x_{r}").shape[0]) for r in RANKS)


def iter_split(loader, model, device):
    """Yield (counts, label, prob_vector) per sample, batch size 1.

    bs=1 keeps per-sample identity trivially correct for every model
    family at the cost of speed; MANTRA forwards are milliseconds.
    """
    model.eval()
    n = len(loader.dataset)
    for i in range(n):
        batch = single_batch(loader, i).to(device)
        batch["model_state"] = "test"
        with torch.no_grad():
            out = model.forward(batch)
            out = model.process_outputs(model_out=out, batch=batch)
        probs = torch.softmax(out["logits"].float(), dim=-1)[0].cpu().numpy()
        label = int(out["labels"].item())
        yield batch_counts(batch), label, probs


# ----------------------------------------------------------------- models --
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def build_menu(pipe, device):
    """Seed-43 val-best published-menu architecture (as-wired selection)."""
    best = None
    for mask, path in menu_paths().items():
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if int(row["seed"]) != SEED:
                continue
            if best is None or row["val_accuracy"] > best[0]["val_accuracy"]:
                best = (row, mask)
    row, mask = best
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, mask)
    ckpt = find_ckpt(
        R / "gccn_menu_mantra" / "runs" / f"{DS}_mask{mask}_s{SEED}"
        / "checkpoints", row.get("selected_epoch"))
    load_ckpt(model, ckpt)
    model.to(device).eval()
    meta = {"mask": mask, "ckpt": ckpt.name,
            "test_balanced_accuracy": row.get("test_balanced_accuracy")}
    return model, meta


def build_gccn(pipe, device):
    """Seed-43 ladder-selected rung (as-wired selection)."""
    lrow = json.loads(gccn_ladder_paths()[SEED].read_text())
    rname = lrow["selected"]
    rung = lrow[rname]
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, rung["mask"])
    ckpt = find_ckpt(
        R / "ladder_b2" / "runs" / f"{DS}_seed{SEED}"
        / f"continue_{rname}" / "checkpoints",
        rung.get("selected_epoch_in_continuation"))
    load_ckpt(model, ckpt)
    model.to(device).eval()
    meta = {"rung": rname, "mask": rung["mask"], "k": rung.get("k"),
            "ckpt": ckpt.name,
            "test_balanced_accuracy": rung.get("test_balanced_accuracy")}
    return model, meta


def build_hopse(device):
    """Seed-43 val-best sweep configuration (as-wired selection)."""
    best = None
    for spec in sweep_configs():
        path = sweep_paths()[spec["tag"]]
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if int(row["seed"]) != SEED:
                continue
            if best is None or row["val_accuracy"] > best[0]["val_accuracy"]:
                best = (row, spec)
    row, spec = best
    pipe = drv.build_pipeline(sweep_compose(
        DS, SEED, spec["members"], SCRATCH / f"attrfig_{spec['tag']}"))
    ckpt = find_ckpt(
        R / "hopse_sweep_mantra" / "runs"
        / f"{DS}_{spec['tag']}_seed{SEED}" / "checkpoints",
        row.get("selected_epoch"))
    load_ckpt(pipe.model, ckpt)
    pipe.model.to(device).eval()
    meta = {"tag": spec["tag"], "ckpt": ckpt.name,
            "test_balanced_accuracy": row.get("test_balanced_accuracy")}
    return pipe, meta


# ------------------------------------------------------------ attribution --
def make_model_fn(model, pos_cls: int, neg_cls: int):
    """Post-encoder stages only (parity-test convention); logit margin."""

    def fn(batch):
        out = model.backbone(batch)
        out = model.readout(model_out=out, batch=batch)
        logits = out["logits"]
        return float(logits[0, pos_cls] - logits[0, neg_cls])

    return fn


def explain_complex(model, batch, pos_cls, neg_cls) -> dict:
    players = [CellPlayer(rank=r, index=i)
               for r in RANKS
               for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
    fn = make_model_fn(model, pos_cls, neg_cls)
    # MANTRA cells have CONSTANT features within each rank (no real
    # per-cell signal), so the original complex_mean baseline replaces
    # every masked row with itself -- a no-op game (verified: all
    # coalition values identical). The informative "signal absent"
    # baseline here is zeros; disclosed in the figure caption.
    raw = CellMaskingGame(fn, batch, players, baseline="zeros",
                          encoder=model.feature_encoder)
    game = CachedGame(n_players=len(players), evaluate=raw)
    full = (1 << len(players)) - 1
    if abs(raw(full) - raw(0)) < 1e-9:
        print("WARNING: degenerate game (v(full)==v(empty))", flush=True)
    att = sampled_shapley(game, len(players), passes=PASSES, seed=0)
    full = (1 << len(players)) - 1
    return {
        "phi": [float(v) for v in att.phi],
        "player_ranks": [p.rank for p in players],
        "v_full": raw(full),
        "v_empty": raw(0),
        "evaluations": game.calls,
        "passes": PASSES,
        "baseline": raw.baseline_name,
    }


def drawing_data(batch) -> dict:
    """Vertex pairs per edge and vertex triples per triangle."""
    inc1 = batch.incidence_1.coalesce()
    inc2 = batch.incidence_2.coalesce()
    v_of_edge = defaultdict(list)
    for (v, e), _ in zip(inc1.indices().t().tolist(), inc1.values()):
        v_of_edge[e].append(v)
    e_of_tri = defaultdict(list)
    for (e, t), _ in zip(inc2.indices().t().tolist(), inc2.values()):
        e_of_tri[t].append(e)
    edges = [sorted(v_of_edge[e]) for e in sorted(v_of_edge)]
    triangles = []
    for t in sorted(e_of_tri):
        verts = sorted({v for e in e_of_tri[t] for v in v_of_edge[e]})
        triangles.append(verts)
    return {"edges": edges, "triangles": triangles}


# ------------------------------------------------------------------ stats --
def count_conditional_auroc(rows) -> tuple[float | None, int]:
    """AUROC restricted to identical-(V,E,F) groups with both classes."""
    groups = defaultdict(list)
    for counts, label, score in rows:
        groups[counts].append((label, score))
    num = den = 0.0
    n_groups = 0
    for g in groups.values():
        pos = [s for (l, s) in g if l == 1]
        neg = [s for (l, s) in g if l == 0]
        if not pos or not neg:
            continue
        n_groups += 1
        for p in pos:
            for q in neg:
                num += 1.0 if p > q else (0.5 if p == q else 0.0)
                den += 1
    return (num / den if den else None), n_groups


def surrogate_agreement(train_rows, test_rows) -> float:
    """Fit logistic (V,E,F) -> model prediction on train; agree on test."""
    from sklearn.linear_model import LogisticRegression

    Xtr = np.array([c for c, _, _ in train_rows], dtype=float)
    ytr = np.array([int(s >= 0.5) for _, _, s in train_rows])
    Xte = np.array([c for c, _, _ in test_rows], dtype=float)
    yte = np.array([int(s >= 0.5) for _, _, s in test_rows])
    if len(set(ytr)) == 1:  # collapsed model: constant surrogate matches
        return float((yte == ytr[0]).mean())
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


# ------------------------------------------------------------------- main --
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = drv.build_pipeline(gccn_compose(SEED, "attrfig"))
    menu_model, menu_meta = build_menu(pipe, device)
    gccn_model, gccn_meta = build_gccn(pipe, device)
    hopse_pipe, hopse_meta = build_hopse(device)

    test_loader = pipe.datamodule.test_dataloader()
    train_loader = pipe.datamodule.train_dataloader()
    h_test_loader = hopse_pipe.datamodule.test_dataloader()
    h_train_loader = hopse_pipe.datamodule.train_dataloader()

    models = {
        "menu": (menu_model, test_loader, train_loader, menu_meta),
        "gccn": (gccn_model, test_loader, train_loader, gccn_meta),
        "hopse": (hopse_pipe.model, h_test_loader, h_train_loader,
                  hopse_meta),
    }

    result = {"seed": SEED, "models": {}}
    split_cache = {}
    pos_cls = neg_cls = None
    for name, (model, te_loader, tr_loader, meta) in models.items():
        print(f"scoring {name} ...", flush=True)
        te_raw = list(iter_split(te_loader, model, device))
        if pos_cls is None:
            # Orientable = majority class index, from the test labels.
            counts_bin = np.bincount([l for _, l, _ in te_raw])
            pos_cls = int(counts_bin.argmax())
            neg_cls = 1 - pos_cls
            result["orientable_class"] = pos_cls
            print(f"orientable class index = {pos_cls} (majority rate "
                  f"{counts_bin.max() / counts_bin.sum():.4f})", flush=True)
        te = [(c, int(l == pos_cls), float(p[pos_cls]))
              for c, l, p in te_raw]
        tr = [(c, int(l == pos_cls), float(p[pos_cls]))
              for c, l, p in iter_split(tr_loader, model, device)]
        cc_auroc, n_groups = count_conditional_auroc(te)
        from sklearn.metrics import roc_auc_score
        plain_auroc = float(roc_auc_score([l for _, l, _ in te],
                                          [s for _, _, s in te]))
        result["models"][name] = {
            "meta": meta,
            "test_auroc": plain_auroc,
            "count_conditional_auroc": cc_auroc,
            "matched_groups": n_groups,
            "surrogate_agreement": surrogate_agreement(tr, te),
            "test_table": [
                {"V": c[0], "E": c[1], "F": c[2], "label": l, "p": s}
                for c, l, s in te],
        }
        split_cache[name] = te
        print(f"  {name}: AUROC {plain_auroc:.3f} | count-cond AUROC "
              f"{cc_auroc:.3f} over {n_groups} groups | surrogate "
              f"{result['models'][name]['surrogate_agreement']:.3f}",
              flush=True)

    # Matched pair: smallest identical-count group with both classes
    # (shared complex indices across the GCCN-family loaders).
    by_counts = defaultdict(lambda: {0: [], 1: []})
    for i, (c, l, _) in enumerate(split_cache["menu"]):
        by_counts[c][l].append(i)
    candidates = sorted(
        (c for c, d in by_counts.items() if d[0] and d[1]),
        key=lambda c: sum(c))
    result["matched_groups_smallest"] = [
        {"counts": list(c),
         "n_orientable": len(by_counts[c][1]),
         "n_non_orientable": len(by_counts[c][0])}
        for c in candidates[:5]]
    pair_counts = candidates[0]
    idx_pos = by_counts[pair_counts][1][0]
    idx_neg = by_counts[pair_counts][0][0]
    # The pair indices come from the GCCN-family loader; the HOPSE
    # datamodule must present the same complexes in the same order.
    for idx in (idx_pos, idx_neg):
        m = split_cache["menu"][idx]
        h = split_cache["hopse"][idx]
        assert m[0] == h[0] and m[1] == h[1], (
            f"test index {idx} differs across datamodules: "
            f"menu {m[:2]} vs hopse {h[:2]}")
    print(f"matched pair: counts {pair_counts}, test indices "
          f"{idx_pos} (orientable) / {idx_neg} (non-orientable)", flush=True)

    pair = {"counts": list(pair_counts)}
    for role, idx in (("orientable", idx_pos), ("non_orientable", idx_neg)):
        entry = {"test_index": idx}
        gbatch = single_batch(test_loader, idx)
        entry["drawing"] = drawing_data(gbatch)
        for name, (model, te_loader, _, _) in models.items():
            batch = single_batch(te_loader, idx).to(device)
            batch["model_state"] = "test"
            print(f"explaining {name} / {role} ...", flush=True)
            entry[name] = explain_complex(model, batch, pos_cls, neg_cls)
            entry[name]["p_orientable"] = split_cache[name][idx][2]
        pair[role] = entry
    result["pair"] = pair

    out_path = OUT / "attr_fig_data.json"
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
