#!/usr/bin/env python
"""LADDER B=2 arm on HOPSE: identical pipeline logic, encoding-space masking.

    python experiments/run_ladder_b2_hopse.py --dataset benzene --seed 42

Mirrors experiments/run_ladder_b2.py exactly, transplanted to HOPSE
(model=cell/hopse_m as #13 ran it, all 8 encoding neighborhoods of
analysis13.json 'references.full_menu.members' in player order):

* Rung choice is FIXED A PRIORI (before any continuation/test results).
* The ladder is backward elimination from the full 8-block coalition
  (mask 255) down to k = 1 on the cheap game, where the cheap game value is
  masked validation accuracy with masking = ENCODING-BLOCK ZEROING (zero the
  dropped neighborhoods' encoding columns at model input — #13's mechanism,
  validated there by an exhaustive 255-coalition column-slicing check).
* Rung 1 = the stop rung (first rung where the best removal would drop v by
  more than the binomial se sqrt(p(1-p)/m), p = v(255), m = val size; floor
  k = 1). Rung 2 = the ladder rung with the highest cheap value among sizes
  different from rung 1's (ties toward the smaller size).
* 'Prune' = fix the coalition's encoding mask permanently (zero dropped
  blocks) and continue training warm from the same fresh 5-epoch stem with
  the dataset's standard TopoBench trainer settings; both rungs continued;
  best-checkpoint validation accuracy selects (tie -> rung 1).
* Honest cost: stem + game + BOTH continuations, plus the once-per-dataset
  encoding preprocessing measured cold on the login node
  (preprocessing_seconds in the JSONL).
* Measured FLOPs per cell (amended protocol, seeds {42,43,44}): stem
  train-epoch, val/test eval passes, and each continued rung's train-epoch
  are profiled inline AFTER all training (FlopCounterMode, #13 conventions;
  see experiments/flops_profiler.py); the encoding-preprocessing FLOPs are
  cited from #13's measured union records
  (data/frozen/hopse13/results/preprocess/pre_<ds>_union.json — identical
  transform on identical data). No estimated quantities.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()  # before building any model

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from flops_profiler import (  # noqa: E402
    HOPSE_CONVENTIONS_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from resume_guard import cell_complete  # noqa: E402
from hopse_masking import (  # noqa: E402
    HOPSE_FULL_MASK,
    HOPSE_N_PLAYERS,
    HOPSE_NEIGHBORHOODS,
    HopseCoalitionMasker,
)
from toposhap.shapley.games import CachedGame  # noqa: E402

DATASETS = ["benzene", "fluoride_carbonyl", "mutagenicity_gxai", "NCI1",
            "MUTAG", "NCI109",  # Tier-1: HOPSE on the TUDataset additions
            "alkane_carbonyl",
            "mantra_orientation",  # MANTRA: simplicial/hopse_m, 6 encodings
            # Citation fill (2026-08-28): node-level task_level exercised
            # through the same pipeline; encoding preprocessing feasibility
            # probed with --prepare-only before submitting arms
            "cocitation_cora", "cocitation_citeseer", "cocitation_pubmed"]

FLOPS_TOTAL_FORMULA = (
    "preprocessing + stem_epochs*(stem_train_epoch + val_pass) + "
    "n_game_evaluations*val_pass + sum over rungs of "
    "[epochs_r*(rung_train_epoch + val_pass) + val_pass + test_pass]"
)

A_PRIORI_RULE = (
    "ladder = backward-elimination path 255->k=1 on the cheap masked-val "
    "game (masking = encoding-block zeroing at model input, #13 mechanism); "
    "rung1 = stop rung (first rung where the best removal would drop v by > "
    "sqrt(p(1-p)/m), p=v(255); floor k=1); rung2 = ladder rung with highest "
    "cheap v among sizes != rung1's k, ties toward smaller size; selection "
    "between trained rungs by best-checkpoint val accuracy, tie -> rung1. "
    "Fixed before any continuation/test results."
)


def hopse_compose(dataset, seed, output_dir):
    """Compose the #13-style HOPSE config (cell/hopse_m, all 8 blocks).

    ``transforms.hopse_encoding.device`` is pinned to cpu so the transform
    parameter hash (and therefore the preprocessing cache) is identical on
    the CPU login node and on GPU compute nodes.
    """
    return drv.compose_config(
        dataset=dataset,
        seed=seed,
        neighborhoods=list(HOPSE_NEIGHBORHOODS),
        output_dir=str(output_dir),
        model=f"{drv.model_domain(dataset)}/hopse_m",
        neighborhood_key="model.preprocessing_params.neighborhoods",
        extra_overrides=["transforms.hopse_encoding.device=cpu"],
    )


def backward_ladder(game, threshold: float):
    """Full elimination ladder plus the a-priori rung-1/rung-2 choice.

    Identical logic to run_ladder_b2.backward_ladder, over 8 players.
    """
    mask = HOPSE_FULL_MASK
    base = game(mask)
    rungs = [{"mask": mask, "k": HOPSE_N_PLAYERS, "cheap_v": float(base)}]
    stop_mask = None
    ties: list[dict] = []
    while bin(mask).count("1") > 1:
        best_bit, best_value = None, -math.inf
        tie_bits: list[int] = []
        for i in range(HOPSE_N_PLAYERS):
            if not mask >> i & 1:
                continue
            value = game(mask & ~(1 << i))
            if value > best_value:
                best_bit, best_value = i, value
                tie_bits = [i]
            elif value == best_value:
                tie_bits.append(i)
        if stop_mask is None and base - best_value > threshold:
            stop_mask = mask  # rung 1 found; keep building the ladder
        if len(tie_bits) > 1:
            ties.append({"k_before": bin(mask).count("1"),
                         "bits": tie_bits, "v": float(best_value)})
        mask &= ~(1 << best_bit)
        rungs.append({
            "mask": mask,
            "k": bin(mask).count("1"),
            "cheap_v": float(best_value),
        })
        base = best_value
    if stop_mask is None:
        stop_mask = mask  # never stopped: floor k = 1

    rung1 = next(r for r in rungs if r["mask"] == stop_mask)
    others = [r for r in rungs if r["k"] != rung1["k"]]
    rung2 = max(others, key=lambda r: (r["cheap_v"], -r["k"]))
    return rungs, dict(rung1), dict(rung2), ties


def mask_members(mask: int) -> list[str]:
    """Neighborhood names in the coalition, bit order."""
    return [
        name for i, name in enumerate(HOPSE_NEIGHBORHOODS) if mask >> i & 1
    ]


def continue_and_test(dataset, seed, model, datamodule, mask, run_dir,
                      device):
    """Fix the encoding mask permanently ('prune'), continue warm, test.

    Returns (record, masker); the masker stays installed so post-hoc FLOPs
    profiling can run under the exact continued configuration.
    """
    masker = HopseCoalitionMasker(model.feature_encoder)
    masker.fix(mask)
    model.to(device)
    cfg = hopse_compose(dataset, seed, run_dir)
    pipe = drv.Pipeline(cfg, datamodule, model, 0.0)
    t0 = time.perf_counter()
    _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
    seconds = time.perf_counter() - t0
    report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
    return {
        "members": mask_members(mask),
        "val": float(report["val_metrics"]["val/accuracy"]),
        "test": float(report["test_metrics"]["test/accuracy"]),
        "continue_seconds": float(seconds),
        "n_epochs_trained_in_continuation": int(epochs),
        "selected_epoch_in_continuation": int(report["selected_epoch"]),
        "n_parameters_backbone": drv.count_parameters(model.backbone),
    }, masker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "ladder_b2_hopse"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prepare-only", action="store_true",
                    help="CPU pre-pass: build + preprocess (encodings) and "
                         "record the cold preprocessing_seconds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = (out_dir
                  / f"ladder_b2_hopse_{args.dataset}_seed{args.seed}.jsonl")
    if not args.prepare_only and cell_complete(final_path):
        print(f"skip: {final_path} already complete")
        return
    run_dir = Path(
        args.run_dir or out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"patches: {PATCHES}")

    stem_cfg = hopse_compose(args.dataset, args.seed, run_dir / "stem")

    prep_path = out_dir / f"preprocessing_time_{args.dataset}.json"
    if args.prepare_only:
        pipe = drv.build_pipeline(stem_cfg, data_only=True)
        prep_path.write_text(json.dumps(
            {args.dataset: pipe.preprocessing_seconds}, indent=1
        ))
        print(f"prepared {args.dataset}: preprocessing "
              f"{pipe.preprocessing_seconds:.1f}s (recorded)")
        return

    preprocessing_seconds = None
    if prep_path.exists():
        preprocessing_seconds = json.loads(prep_path.read_text()).get(
            args.dataset
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- stem (fresh, 5 epochs, no early stop) -----------------------------
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    # ---- cheap game (encoding-block zeroing) + full ladder ------------------
    masker = HopseCoalitionMasker(model.feature_encoder)
    val_loader = pipe.datamodule.val_dataloader()
    m_val = {}

    def evaluate(mask: int) -> float:
        with masker.coalition(mask):
            acc, n = drv.accuracy_over_loader(
                model, val_loader, "Validation", device
            )
        m_val["n"] = n
        return acc

    game = CachedGame(n_players=HOPSE_N_PLAYERS, evaluate=evaluate)
    t0 = time.perf_counter()
    v_full = game(HOPSE_FULL_MASK)
    binomial_se = math.sqrt(v_full * (1.0 - v_full) / m_val["n"])
    rungs, rung1, rung2, ties = backward_ladder(game, threshold=binomial_se)
    game_seconds = time.perf_counter() - t0
    print(
        f"ladder rung1: mask={rung1['mask']} k={rung1['k']} "
        f"cheap_v={rung1['cheap_v']:.4f}; rung2: mask={rung2['mask']} "
        f"k={rung2['k']} cheap_v={rung2['cheap_v']:.4f}; "
        f"se={binomial_se:.4f} evals={game.calls}"
    )

    # ---- both rungs warm from the SAME stem --------------------------------
    masker.restore()  # unhook the game before snapshotting
    model_r2 = copy.deepcopy(model)  # pristine stem snapshot

    rung1_rec, masker1 = continue_and_test(
        args.dataset, args.seed, model, pipe.datamodule,
        rung1["mask"], run_dir / "continue_rung1", device,
    )
    rung1.update(rung1_rec)
    rung2_rec, masker2 = continue_and_test(
        args.dataset, args.seed, model_r2, pipe.datamodule,
        rung2["mask"], run_dir / "continue_rung2", device,
    )
    rung2.update(rung2_rec)

    # ---- select by validation (tie -> rung1) -------------------------------
    selected = "rung2" if rung2["val"] > rung1["val"] else "rung1"
    winner = rung1 if selected == "rung1" else rung2

    # ---- measured FLOPs (after ALL training; cannot perturb RNG streams) ---
    train_loader = pipe.datamodule.train_dataloader()
    test_loader = pipe.datamodule.test_dataloader()
    with masker1.coalition(HOPSE_FULL_MASK):  # stem = full coalition
        stem_train_prof = profile_train_epoch(model, train_loader, device)
        val_pass_prof = profile_eval_pass(model, val_loader, device)
        test_pass_prof = profile_eval_pass(
            model, test_loader, device, "Test"
        )
    rung1_train_prof = profile_train_epoch(model, train_loader, device)
    rung2_train_prof = profile_train_epoch(model_r2, train_loader, device)

    union_path = (REPO / "data" / "frozen" / "hopse13" / "results"
                  / "preprocess" / f"pre_{args.dataset}_union.json")
    if union_path.exists():
        union = json.loads(union_path.read_text())
        preprocessing_flops = int(union["flops"]["flops"])
        preproc_flops_source = (
            f"data/frozen/hopse13/results/preprocess/pre_{args.dataset}"
            "_union.json (measured, identical transform/data; "
            "once-per-dataset, uncharged amortization left downstream)"
        )
    else:
        # Tier-1 datasets have no frozen #13 interceptor measurement; the
        # clocked preprocessing_seconds is recorded instead — no estimates.
        preprocessing_flops = None
        preproc_flops_source = (
            "no frozen measured record for this dataset; excluded from "
            "total_flops (preprocessing_seconds clocked instead)"
        )

    e1 = rung1["n_epochs_trained_in_continuation"]
    e2 = rung2["n_epochs_trained_in_continuation"]
    vp, tp = val_pass_prof["flops"], test_pass_prof["flops"]
    flops_total = (
        (preprocessing_flops or 0)
        + args.stem_epochs * (stem_train_prof["flops"] + vp)
        + game.calls * vp
        + e1 * (rung1_train_prof["flops"] + vp) + vp + tp
        + e2 * (rung2_train_prof["flops"] + vp) + vp + tp
    )
    flops = {
        "conventions": HOPSE_CONVENTIONS_NOTE,
        "total_formula": FLOPS_TOTAL_FORMULA,
        "preprocessing_flops": preprocessing_flops,
        "preprocessing_flops_source": preproc_flops_source,
        "stem_train_epoch": stem_train_prof,
        "val_pass": val_pass_prof,
        "test_pass": test_pass_prof,
        "rung1_train_epoch": rung1_train_prof,
        "rung2_train_epoch": rung2_train_prof,
        "n_epochs_rung1": e1,
        "n_epochs_rung2": e2,
        "total_flops": int(flops_total),
    }
    print(f"flops: total {flops_total:.3e} (stem epoch "
          f"{stem_train_prof['flops']:.3e}, val pass {vp:.3e})")

    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selector": "ladder_b2_hopse",
        "meta_a_priori_rule": A_PRIORI_RULE,
        "players": list(HOPSE_NEIGHBORHOODS),
        "rung1": rung1,
        "rung2": rung2,
        "selected": selected,
        "mask": int(winner["mask"]),
        "k": int(winner["k"]),
        "val_accuracy": float(winner["val"]),
        "test_accuracy": float(winner["test"]),
        "ladder": rungs,
        "ties": ties,
        "v_full": float(v_full),
        "binomial_se": float(binomial_se),
        "m_val": int(m_val["n"]),
        "monitored_metric": str(stem_cfg.dataset.parameters.monitor_metric),
        "floor_metric": "accuracy",
        "floor_note": "game values and the binomial-se floor use plain val accuracy; checkpointing/early-stop monitor the dataset's standard metric (conservative stand-in, disclosed, when they differ)",
        "stem_seconds": float(stem_seconds),
        "game_seconds": float(game_seconds),
        "n_game_evaluations": int(game.calls),
        "continue_seconds_rung1": rung1["continue_seconds"],
        "continue_seconds_rung2": rung2["continue_seconds"],
        "n_epochs_trained_in_continuation_rung1":
            rung1["n_epochs_trained_in_continuation"],
        "n_epochs_trained_in_continuation_rung2":
            rung2["n_epochs_trained_in_continuation"],
        "preprocessing_seconds": preprocessing_seconds,
        "preprocessing_note": "cold once-per-dataset login-node CPU "
                              "pre-pass, shared across seeds",
        "flops": flops,
        "n_parameters_total": drv.count_parameters(model),
        "stem_epochs": int(args.stem_epochs),
        "model": "hopse_m",
        "orientation": "src_to_dst",
        "patches": PATCHES,
    }
    out_path = out_dir / f"ladder_b2_hopse_{args.dataset}_seed{args.seed}.jsonl"
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}")
    print(
        f"selected={selected} k={row['k']} mask={row['mask']} "
        f"val={row['val_accuracy']:.4f} test={row['test_accuracy']:.4f} "
        f"wall={stem_seconds + game_seconds + rung1['continue_seconds'] + rung2['continue_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
