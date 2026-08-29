#!/usr/bin/env python
"""Balanced-accuracy backfill for every completed MANTRA cell (eval-only).

    python experiments/mantra_balanced_eval.py            # patch all pending
    python experiments/mantra_balanced_eval.py --check    # exit 2 if pending

MANTRA orientability is heavily class-imbalanced: plain accuracy cannot
adjudicate the cross-family contrast (a majority-collapsed classifier sits
at the majority rate, and the historical counting ceiling 0.5054 was
BALANCED accuracy). This script re-evaluates each cell's saved best
checkpoint on val and test (cheap eval-only passes) and patches
``val_balanced_accuracy`` / ``test_balanced_accuracy`` into the JSONLs in
place — add-only, existing fields are never overwritten. It also records the
recomputed plain accuracy in a ``balanced_eval`` sub-dict as a load-fidelity
cross-check against the recorded value.

Selection integrity (for the record): the arms SELECTED on plain val
accuracy exactly as wired; balanced metrics are recorded alongside and are
NEVER used to re-select. Ladder rows get both rungs patched; the top-level
balanced fields are the AS-WIRED selected rung's.

Also writes results/mantra_prepare_logs/class_balance.json: per-seed,
per-split class counts and majority rates for the orientation task.

Idempotent and resumable: each JSONL is patched as soon as its cells are
evaluated; rows already carrying balanced fields are skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()  # before building any model

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from hopse_masking import HopseCoalitionMasker  # noqa: E402
from run_gccn_menu import MENU_MASKS  # noqa: E402
from run_hopse_sweep import compose as sweep_compose  # noqa: E402
from run_hopse_sweep import sweep_configs  # noqa: E402
from run_ladder_b2_hopse import hopse_compose  # noqa: E402
from toposhap.neighborhoods import prune_backbone_  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS, mask_to_coalition  # noqa: E402

DS = "mantra_orientation"
SEEDS = [42, 43, 44]
R = REPO / "results"
SCRATCH = R / "balanced_eval_runs"

NOTE = ("balanced accuracy = mean per-class recall, argmax preds, best "
        "checkpoint reloaded; selection used plain val accuracy as wired "
        "(balanced never re-selects)")


# ---------------------------------------------------------------- metrics --
@torch.no_grad()
def split_metrics(model, loader, state: str, device) -> dict:
    """Plain + balanced accuracy and per-class counts over a loader."""
    was_training = model.training
    model.eval()
    model.state_str = state
    per_total: dict[int, int] = {}
    per_correct: dict[int, int] = {}
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        batch["model_state"] = state
        out = model.forward(batch)
        out = model.process_outputs(model_out=out, batch=batch)
        preds = out["logits"].argmax(dim=-1)
        labels = out["labels"]
        import torch as _t
        all_probs.append(_t.softmax(out["logits"].float(), dim=-1)[:, -1]
                         .detach().cpu())
        all_labels.append(labels.detach().cpu())
        for c in labels.unique().tolist():
            c = int(c)
            m = labels == c
            per_total[c] = per_total.get(c, 0) + int(m.sum())
            per_correct[c] = per_correct.get(c, 0) + int((preds[m] == c).sum())
    if was_training:
        model.train()
    total = sum(per_total.values())
    plain = sum(per_correct.values()) / total
    balanced = sum(per_correct[c] / per_total[c] for c in per_total) / len(
        per_total)
    try:
        from sklearn.metrics import roc_auc_score
        import torch as _t
        auroc = float(roc_auc_score(_t.cat(all_labels).numpy(),
                                    _t.cat(all_probs).numpy()))
    except Exception:
        auroc = None
    return {"plain": float(plain), "balanced": float(balanced),
            "auroc": auroc,
            "per_class_total": per_total, "per_class_correct": per_correct}


def eval_both(model, datamodule, device) -> tuple[dict, dict]:
    val = split_metrics(model, datamodule.val_dataloader(), "Validation",
                        device)
    test = split_metrics(model, datamodule.test_dataloader(), "Test", device)
    return val, test


def find_ckpt(ckpt_dir: Path, selected_epoch: int | None = None) -> Path:
    cands = sorted(Path(ckpt_dir).glob("epoch_*.ckpt"))
    if selected_epoch is not None:
        hit = [c for c in cands
               if int(re.search(r"epoch_(\d+)", c.name).group(1))
               == selected_epoch]
        if hit:
            return hit[0]
    if len(cands) != 1:
        raise FileNotFoundError(
            f"{ckpt_dir}: expected one epoch_*.ckpt (or one matching "
            f"selected epoch {selected_epoch}), found {cands}")
    return cands[0]


def load_ckpt(model, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)


def recheck(recorded: float, recomputed: float, what: str) -> None:
    if abs(recorded - recomputed) > 1e-3:
        print(f"  WARNING {what}: recorded plain {recorded:.6f} != "
              f"recomputed {recomputed:.6f} — checkpoint/rebuild mismatch?")


def patch(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n")
    print(f"patched {path}")


def patch_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"patched {path}")


# ------------------------------------------------------------- inventory --
def gccn_ladder_paths():
    return {s: R / "ladder_b2" / f"ladder_b2_{DS}_seed{s}.jsonl"
            for s in SEEDS}


def hopse_ladder_paths():
    return {s: R / "ladder_b2_hopse" / f"ladder_b2_hopse_{DS}_seed{s}.jsonl"
            for s in SEEDS}


def anchored_paths():
    return {s: R / "anchored3_exact" / f"anchored3_{DS}_seed{s}.jsonl"
            for s in SEEDS}


def k3_paths():
    return {s: R / "hopse_k3_rung" / f"k3_{DS}_seed{s}.jsonl" for s in SEEDS}


def sweep_paths():
    return {c["tag"]: R / "hopse_sweep_mantra" / f"sweep_{DS}_{c['tag']}.jsonl"
            for c in sweep_configs()}


def menu_paths():
    return {m: R / "gccn_menu_mantra" / f"menu_{DS}_mask{m}.jsonl"
            for m in MENU_MASKS}


def row_done(row: dict) -> bool:
    return ("val_balanced_accuracy" in row
            and "test_balanced_accuracy" in row
            and "test_auroc" in row)


def ladder_row_done(row: dict) -> bool:
    return (row_done(row) and row_done(row["rung1"])
            and row_done(row["rung2"]))


def pending_report() -> list[str]:
    """Existing-but-unpatched cells (missing cells are the arrays' job)."""
    pending = []
    for name, paths, done in [
        ("gccn_ladder", gccn_ladder_paths(), ladder_row_done),
        ("hopse_ladder", hopse_ladder_paths(), ladder_row_done),
        ("anchored3", anchored_paths(), row_done),
        ("k3_rung", k3_paths(), row_done),
    ]:
        for key, p in paths.items():
            if p.exists() and not done(json.loads(p.read_text())):
                pending.append(f"{name}:{key}")
    for name, paths in [("sweep", sweep_paths()), ("menu", menu_paths())]:
        for key, p in paths.items():
            if not p.exists():
                continue
            rows = [json.loads(x) for x in p.read_text().splitlines() if x]
            if not all(row_done(r) for r in rows):
                pending.append(f"{name}:{key}")
    if not (R / "mantra_prepare_logs" / "class_balance.json").exists():
        pending.append("class_balance.json")
    return pending


# ------------------------------------------------------------- families ---
def gccn_compose(seed: int, tag: str):
    return drv.compose_config(
        dataset=DS, seed=seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(SCRATCH / f"gccn_{tag}_seed{seed}"),
        model=f"{drv.model_domain(DS)}/{drv.RECORD_MODEL.split('/', 1)[1]}",
    )


def eval_gccn_pruned(pipe, mask: int, ckpt: Path, device) -> tuple[dict, dict]:
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, mask)
    load_ckpt(model, ckpt)
    model.to(device)
    return eval_both(model, pipe.datamodule, device)


def do_gccn_family(device) -> None:
    """GCCN ladder rungs + anchored arm (shared full-vocabulary pipeline)."""
    for seed in SEEDS:
        lpath = gccn_ladder_paths()[seed]
        apath = anchored_paths()[seed]
        lrow = json.loads(lpath.read_text()) if lpath.exists() else None
        arow = json.loads(apath.read_text()) if apath.exists() else None
        need_l = lrow is not None and not ladder_row_done(lrow)
        need_a = arow is not None and not row_done(arow)
        if not need_l and not need_a:
            continue
        pipe = drv.build_pipeline(gccn_compose(seed, "ladder"))
        if need_l:
            run_dir = R / "ladder_b2" / "runs" / f"{DS}_seed{seed}"
            for rname in ("rung1", "rung2"):
                rung = lrow[rname]
                if row_done(rung):
                    continue
                ckpt = find_ckpt(
                    run_dir / f"continue_{rname}" / "checkpoints",
                    rung.get("selected_epoch_in_continuation"))
                val, test = eval_gccn_pruned(pipe, rung["mask"], ckpt, device)
                recheck(rung["val"], val["plain"],
                        f"gccn ladder s{seed} {rname} val")
                recheck(rung["test"], test["plain"],
                        f"gccn ladder s{seed} {rname} test")
                rung["val_balanced_accuracy"] = val["balanced"]
                rung["test_balanced_accuracy"] = test["balanced"]
                rung["val_auroc"] = val["auroc"]
                rung["test_auroc"] = test["auroc"]
                rung["balanced_eval"] = {
                    "note": NOTE, "ckpt": ckpt.name,
                    "plain_val_recheck": val["plain"],
                    "plain_test_recheck": test["plain"]}
            winner = lrow[lrow["selected"]]
            lrow.setdefault("val_balanced_accuracy",
                            winner["val_balanced_accuracy"])
            lrow.setdefault("test_balanced_accuracy",
                            winner["test_balanced_accuracy"])
            lrow.setdefault("val_auroc", winner.get("val_auroc"))
            lrow.setdefault("test_auroc", winner.get("test_auroc"))
            patch(lpath, lrow)
        if need_a:
            ckpt = find_ckpt(
                R / "anchored3_exact" / "runs" / f"{DS}_seed{seed}"
                / "continue" / "checkpoints",
                arow.get("selected_epoch_in_continuation"))
            val, test = eval_gccn_pruned(pipe, arow["mask"], ckpt, device)
            recheck(arow["val_accuracy"], val["plain"],
                    f"anchored3 s{seed} val")
            recheck(arow["test_accuracy"], test["plain"],
                    f"anchored3 s{seed} test")
            arow["val_balanced_accuracy"] = val["balanced"]
            arow["test_balanced_accuracy"] = test["balanced"]
            arow["val_auroc"] = val["auroc"]
            arow["test_auroc"] = test["auroc"]
            arow["balanced_eval"] = {
                "note": NOTE, "ckpt": ckpt.name,
                "plain_val_recheck": val["plain"],
                "plain_test_recheck": test["plain"]}
            patch(apath, arow)


def do_hopse_family(device) -> None:
    """HOPSE ladder rungs + the k3 (ladder rung) arm (shared pipeline)."""
    for seed in SEEDS:
        lpath = hopse_ladder_paths()[seed]
        kpath = k3_paths()[seed]
        lrow = json.loads(lpath.read_text()) if lpath.exists() else None
        krow = json.loads(kpath.read_text()) if kpath.exists() else None
        need_l = lrow is not None and not ladder_row_done(lrow)
        need_k = krow is not None and not row_done(krow)
        if not need_l and not need_k:
            continue
        pipe = drv.build_pipeline(
            hopse_compose(DS, seed, SCRATCH / f"hopse_seed{seed}"))
        masker = HopseCoalitionMasker(pipe.model.feature_encoder)
        if need_l:
            run_dir = R / "ladder_b2_hopse" / "runs" / f"{DS}_seed{seed}"
            for rname in ("rung1", "rung2"):
                rung = lrow[rname]
                if row_done(rung):
                    continue
                ckpt = find_ckpt(
                    run_dir / f"continue_{rname}" / "checkpoints",
                    rung.get("selected_epoch_in_continuation"))
                load_ckpt(pipe.model, ckpt)
                pipe.model.to(device)
                masker.fix(rung["mask"])
                val, test = eval_both(pipe.model, pipe.datamodule, device)
                recheck(rung["val"], val["plain"],
                        f"hopse ladder s{seed} {rname} val")
                recheck(rung["test"], test["plain"],
                        f"hopse ladder s{seed} {rname} test")
                rung["val_balanced_accuracy"] = val["balanced"]
                rung["test_balanced_accuracy"] = test["balanced"]
                rung["val_auroc"] = val["auroc"]
                rung["test_auroc"] = test["auroc"]
                rung["balanced_eval"] = {
                    "note": NOTE, "ckpt": ckpt.name,
                    "plain_val_recheck": val["plain"],
                    "plain_test_recheck": test["plain"]}
            winner = lrow[lrow["selected"]]
            lrow.setdefault("val_balanced_accuracy",
                            winner["val_balanced_accuracy"])
            lrow.setdefault("test_balanced_accuracy",
                            winner["test_balanced_accuracy"])
            lrow.setdefault("val_auroc", winner.get("val_auroc"))
            lrow.setdefault("test_auroc", winner.get("test_auroc"))
            patch(lpath, lrow)
        if need_k:
            ckpt = find_ckpt(
                R / "hopse_k3_rung" / "runs" / f"{DS}_seed{seed}"
                / "continue" / "checkpoints", krow.get("selected_epoch"))
            load_ckpt(pipe.model, ckpt)
            pipe.model.to(device)
            masker.fix(krow["mask"])
            val, test = eval_both(pipe.model, pipe.datamodule, device)
            recheck(krow["val_accuracy"], val["plain"], f"k3 s{seed} val")
            recheck(krow["test_accuracy"], test["plain"], f"k3 s{seed} test")
            krow["val_balanced_accuracy"] = val["balanced"]
            krow["test_balanced_accuracy"] = test["balanced"]
            krow["val_auroc"] = val["auroc"]
            krow["test_auroc"] = test["auroc"]
            krow["balanced_eval"] = {
                "note": NOTE, "ckpt": ckpt.name,
                "plain_val_recheck": val["plain"],
                "plain_test_recheck": test["plain"]}
            patch(kpath, krow)
        masker.restore()


def do_sweep(device) -> None:
    for spec in sweep_configs():
        tag, members = spec["tag"], spec["members"]
        path = sweep_paths()[tag]
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        if all(row_done(r) for r in rows):
            continue
        for row in rows:
            if row_done(row):
                continue
            seed = int(row["seed"])
            pipe = drv.build_pipeline(sweep_compose(
                DS, seed, members, SCRATCH / f"sweep_{tag}_seed{seed}"))
            ckpt = find_ckpt(
                R / "hopse_sweep_mantra" / "runs"
                / f"{DS}_{tag}_seed{seed}" / "checkpoints",
                row.get("selected_epoch"))
            load_ckpt(pipe.model, ckpt)
            pipe.model.to(device)
            val, test = eval_both(pipe.model, pipe.datamodule, device)
            recheck(row["val_accuracy"], val["plain"],
                    f"sweep {tag} s{seed} val")
            recheck(row["test_accuracy"], test["plain"],
                    f"sweep {tag} s{seed} test")
            row["val_balanced_accuracy"] = val["balanced"]
            row["test_balanced_accuracy"] = test["balanced"]
            row["val_auroc"] = val["auroc"]
            row["test_auroc"] = test["auroc"]
            row["balanced_eval"] = {
                "note": NOTE, "ckpt": ckpt.name,
                "plain_val_recheck": val["plain"],
                "plain_test_recheck": test["plain"]}
        patch_lines(path, rows)


def do_menu(device) -> None:
    for mask in MENU_MASKS:
        path = menu_paths()[mask]
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        if all(row_done(r) for r in rows):
            continue
        members = list(mask_to_coalition(mask))
        for row in rows:
            if row_done(row):
                continue
            seed = int(row["seed"])
            pipe = drv.build_pipeline(drv.compose_config(
                dataset=DS, seed=seed, neighborhoods=members,
                output_dir=str(SCRATCH / f"menu_{mask}_seed{seed}"),
                model=f"{drv.model_domain(DS)}/"
                      f"{drv.RECORD_MODEL.split('/', 1)[1]}"))
            ckpt = find_ckpt(
                R / "gccn_menu_mantra" / "runs"
                / f"{DS}_mask{mask}_s{seed}" / "checkpoints",
                row.get("selected_epoch"))
            load_ckpt(pipe.model, ckpt)
            pipe.model.to(device)
            val, test = eval_both(pipe.model, pipe.datamodule, device)
            recheck(row["val_accuracy"], val["plain"],
                    f"menu {mask} s{seed} val")
            recheck(row["test_accuracy"], test["plain"],
                    f"menu {mask} s{seed} test")
            row["val_balanced_accuracy"] = val["balanced"]
            row["test_balanced_accuracy"] = test["balanced"]
            row["val_auroc"] = val["auroc"]
            row["test_auroc"] = test["auroc"]
            row["balanced_eval"] = {
                "note": NOTE, "ckpt": ckpt.name,
                "plain_val_recheck": val["plain"],
                "plain_test_recheck": test["plain"]}
        patch_lines(path, rows)


def do_class_balance() -> None:
    out = R / "mantra_prepare_logs" / "class_balance.json"
    if out.exists():
        return
    record: dict = {"dataset": DS, "task": "orientation",
                    "note": "per-seed per-split class counts; majority_rate "
                            "= max class share (the collapsed-classifier "
                            "reference line)"}
    for seed in SEEDS:
        pipe = drv.build_pipeline(
            hopse_compose(DS, seed, SCRATCH / f"cb_seed{seed}"),
            data_only=True)
        per_seed = {}
        for split, loader in [
            ("train", pipe.datamodule.train_dataloader()),
            ("val", pipe.datamodule.val_dataloader()),
            ("test", pipe.datamodule.test_dataloader()),
        ]:
            counts: dict[int, int] = {}
            for batch in loader:
                for c in batch.y.flatten().tolist():
                    counts[int(c)] = counts.get(int(c), 0) + 1
            n = sum(counts.values())
            per_seed[split] = {
                "counts": {str(k): v for k, v in sorted(counts.items())},
                "n": n,
                "majority_rate": max(counts.values()) / n,
            }
        record[f"seed{seed}"] = per_seed
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 2 if any existing cell lacks balanced fields")
    args = ap.parse_args()

    pending = pending_report()
    if args.check:
        if pending:
            print(f"pending balanced eval ({len(pending)}): {pending}")
            sys.exit(2)
        print("balanced eval complete for all existing MANTRA cells")
        return
    if not pending:
        print("nothing pending")
        return
    print(f"pending ({len(pending)}): {pending}")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_class_balance()
    do_gccn_family(device)
    do_hopse_family(device)
    do_sweep(device)
    do_menu(device)

    left = pending_report()
    print(f"remaining pending after pass: {left if left else 'none'}")


if __name__ == "__main__":
    main()
