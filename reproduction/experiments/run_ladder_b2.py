#!/usr/bin/env python
"""LADDER B=2 arm: two a-priori rungs off one stem, selected by validation.

    python experiments/run_ladder_b2.py --dataset benzene --seed 42

Rung choice is FIXED A PRIORI (before any continuation/test results):

* The ladder is the backward-elimination path from the full coalition
  (mask 2047) down to k = 1: at each step remove the member whose removal
  hurts the cheap masked-validation game least (ties toward the lowest bit).
* Rung 1 = the stop rung — the smallest rung where the next removal would
  drop the cheap value by MORE than the binomial se sqrt(p(1-p)/m),
  p = v(2047), m = val size (identical to the autok_backward arm; floor
  k = 1 when elimination never triggers the stop).
* Rung 2 = the ladder rung with the HIGHEST cheap game value among sizes
  different from rung 1's (ties toward the smaller size).

Both rungs are pruned warm from the SAME fresh 5-epoch stem and continued
with the dataset's standard TopoBench trainer settings; the reported
test_accuracy is the rung selected by best-checkpoint validation accuracy
(tie -> rung 1). Wall cost is honest B=2: stem + game + BOTH continuations.

Protocol otherwise identical to run_autok.py: fixed interrank orientation
patches before any model, cell/topotune_nonlinear_exact, full 11-vocabulary
in canonical bit order.
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
    GCCN_COUNTER_NOTE,
    profile_eval_pass,
    profile_train_epoch,
)
from resume_guard import cell_complete  # noqa: E402
from run_autok import DATASETS  # noqa: E402
from toposhap.neighborhoods import (  # noqa: E402
    CoalitionMaskedBackbone,
    prune_backbone_,
)
from toposhap.shapley.games import CachedGame  # noqa: E402

FLOPS_TOTAL_FORMULA = (
    "stem_epochs*(stem_train_epoch + val_pass) + "
    "n_game_evaluations*val_pass + sum over rungs of "
    "[epochs_r*(rung_train_epoch + val_pass) + val_pass + test_pass]; "
    "lifting preprocessing excluded, matching #14's per-run accounting"
)
from toposhap.vocabulary import (  # noqa: E402
    FULL_MASK,
    N_PLAYERS,
    NEIGHBORHOODS,
)

A_PRIORI_RULE = (
    "ladder = backward-elimination path 2047->k=1 on the cheap masked-val "
    "game; rung1 = stop rung (first rung where the best removal would drop "
    "v by > sqrt(p(1-p)/m), p=v(2047); floor k=1); rung2 = ladder rung with "
    "highest cheap v among sizes != rung1's k, ties toward smaller size; "
    "selection between trained rungs by best-checkpoint val accuracy, tie "
    "-> rung1. Fixed before any continuation/test results."
)


def backward_ladder(game, threshold: float):
    """Full elimination ladder plus the a-priori rung-1/rung-2 choice.

    Returns (rungs, rung1, rung2, ties): rungs is the list of
    {mask, k, cheap_v} down the path (size 11 first, size 1 last); rung1 is
    the stop rung; rung2 the highest-cheap-v rung of a different size.
    """
    mask = FULL_MASK
    base = game(mask)
    rungs = [{"mask": mask, "k": N_PLAYERS, "cheap_v": float(base)}]
    stop_mask = None
    ties: list[dict] = []
    while bin(mask).count("1") > 1:
        best_bit, best_value = None, -math.inf
        tie_bits: list[int] = []
        for i in range(N_PLAYERS):
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
    # highest cheap value; ties toward the smaller size
    rung2 = max(others, key=lambda r: (r["cheap_v"], -r["k"]))
    return rungs, dict(rung1), dict(rung2), ties


def continue_and_test(dataset, seed, model, datamodule, mask, run_dir,
                      device, model_name=None):
    """Warm-prune a model copy to `mask`, continue, test best-val ckpt."""
    inner = model.backbone.backbone
    kept = prune_backbone_(inner, mask)
    model.to(device)
    cfg = drv.compose_config(
        dataset=dataset,
        seed=seed,
        neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(run_dir),
        model=model_name or drv.RECORD_MODEL,
    )
    pipe = drv.Pipeline(cfg, datamodule, model, 0.0)
    t0 = time.perf_counter()
    _, ckpt_cb, _, epochs = drv.fit_continuation(cfg, pipe)
    seconds = time.perf_counter() - t0
    report = drv.test_best_checkpoint(cfg, pipe, ckpt_cb, device)
    return {
        "members": kept,
        "val": float(report["val_metrics"]["val/accuracy"]),
        "test": float(report["test_metrics"]["test/accuracy"]),
        "continue_seconds": float(seconds),
        "n_epochs_trained_in_continuation": int(epochs),
        "selected_epoch_in_continuation": int(report["selected_epoch"]),
        "n_parameters_backbone": drv.count_parameters(inner),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir", default=str(REPO / "results" / "ladder_b2"))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"ladder_b2_{args.dataset}_seed{args.seed}.jsonl"
    if cell_complete(final_path):
        print(f"skip: {final_path} already complete")
        return
    run_dir = Path(
        args.run_dir or out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"patches: {PATCHES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- stem (fresh, 5 epochs, no early stop) -----------------------------
    stem_cfg = drv.compose_config(
        dataset=args.dataset,
        seed=args.seed,
        neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(run_dir / "stem"),
        model=f"{drv.model_domain(args.dataset)}/{args.model}",
    )
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    # ---- cheap game + full ladder ------------------------------------------
    inner = model.backbone.backbone
    wrapper = CoalitionMaskedBackbone(inner)
    val_loader = pipe.datamodule.val_dataloader()
    m_val = {}

    def evaluate(mask: int) -> float:
        with wrapper.coalition(mask):
            acc, n = drv.accuracy_over_loader(
                model, val_loader, "Validation", device
            )
        m_val["n"] = n
        return acc

    game = CachedGame(n_players=N_PLAYERS, evaluate=evaluate)
    t0 = time.perf_counter()
    v_full = game(FULL_MASK)
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
    inner.aggregate_inter_nbhd = wrapper._original_aggregate  # unhook game
    stem_snapshot = copy.deepcopy(model)  # full architecture, for profiling
    model_r2 = copy.deepcopy(model)  # snapshot stem before rung-1 pruning

    model_name = f"{drv.model_domain(args.dataset)}/{args.model}"
    rung1.update(continue_and_test(
        args.dataset, args.seed, model, pipe.datamodule,
        rung1["mask"], run_dir / "continue_rung1", device,
        model_name=model_name,
    ))
    rung2.update(continue_and_test(
        args.dataset, args.seed, model_r2, pipe.datamodule,
        rung2["mask"], run_dir / "continue_rung2", device,
        model_name=model_name,
    ))

    # ---- select by validation (tie -> rung1) -------------------------------
    selected = "rung2" if rung2["val"] > rung1["val"] else "rung1"
    winner = rung1 if selected == "rung1" else rung2

    # ---- measured FLOPs (after ALL training; cannot perturb RNG streams) ---
    train_loader = pipe.datamodule.train_dataloader()
    test_loader = pipe.datamodule.test_dataloader()
    stem_snapshot.to(device)
    stem_train_prof = profile_train_epoch(stem_snapshot, train_loader, device)
    val_pass_prof = profile_eval_pass(stem_snapshot, val_loader, device)
    test_pass_prof = profile_eval_pass(
        stem_snapshot, test_loader, device, "Test"
    )
    rung1_train_prof = profile_train_epoch(model, train_loader, device)
    rung2_train_prof = profile_train_epoch(model_r2, train_loader, device)

    e1 = rung1["n_epochs_trained_in_continuation"]
    e2 = rung2["n_epochs_trained_in_continuation"]
    vp, tp = val_pass_prof["flops"], test_pass_prof["flops"]
    flops_total = (
        args.stem_epochs * (stem_train_prof["flops"] + vp)
        + game.calls * vp
        + e1 * (rung1_train_prof["flops"] + vp) + vp + tp
        + e2 * (rung2_train_prof["flops"] + vp) + vp + tp
    )
    flops = {
        "conventions": GCCN_COUNTER_NOTE,
        "total_formula": FLOPS_TOTAL_FORMULA,
        "note": "eval passes profiled on the full stem architecture; rung "
                "train-epochs on the pruned models",
        "stem_train_epoch": stem_train_prof,
        "val_pass": val_pass_prof,
        "test_pass": test_pass_prof,
        "rung1_train_epoch": rung1_train_prof,
        "rung2_train_epoch": rung2_train_prof,
        "n_epochs_rung1": e1,
        "n_epochs_rung2": e2,
        "total_flops": int(flops_total),
    }
    print(f"flops: total {flops_total:.3e}")

    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selector": "ladder_b2",
        "meta_a_priori_rule": A_PRIORI_RULE,
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
        "flops": flops,
        "stem_seconds": float(stem_seconds),
        "game_seconds": float(game_seconds),
        "n_game_evaluations": int(game.calls),
        "continue_seconds_rung1": rung1["continue_seconds"],
        "continue_seconds_rung2": rung2["continue_seconds"],
        "n_epochs_trained_in_continuation_rung1":
            rung1["n_epochs_trained_in_continuation"],
        "n_epochs_trained_in_continuation_rung2":
            rung2["n_epochs_trained_in_continuation"],
        "stem_epochs": int(args.stem_epochs),
        "model": args.model,
        "orientation": "src_to_dst",
        "patches": PATCHES,
    }
    out_path = out_dir / f"ladder_b2_{args.dataset}_seed{args.seed}.jsonl"
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
