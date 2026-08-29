#!/usr/bin/env python
"""Anchored k=3 arm, measurement-exact: production sampled game + real FLOPs.

    python experiments/run_anchored3.py --dataset benzene --seed 42

Seeds {42,43,44} x 7 datasets. Per cell:

1. Fixed-orientation patches, then the 5-epoch full-11-vocabulary stem
   (cell/topotune_nonlinear_exact — the config that reproduces the #14
   parameter record; L.seed_everything).
2. Sampled Shapley game at the production budget: 128 passes, antithetic
   permutation sampling, unique-evaluation accounting via CachedGame.calls;
   game value = masked full-val-set accuracy through CoalitionMaskedBackbone
   on the inner TopoTune module.
3. Anchored top-3 selection (anchor = up_adjacency-0, bit 0).
4. Warm prune (prune_backbone_) + continuation with the dataset's standard
   trainer settings + best-val-checkpoint test via the normal path.
5. Measured FLOPs, profiled inline AFTER all training (#14 conventions,
   FlopCounterMode, backward through a scalar sum of logits; sparse scatter
   uncounted; lifting preprocessing excluded, matching #14's per-run
   accounting): stem train-epoch, val/test eval passes, continued-mask
   train-epoch, actual epochs trained. No estimated quantities.
"""

from __future__ import annotations

import argparse
import copy
import json
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
from toposhap.selection.selectors import anchored_top_k_mask  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402
from toposhap.vocabulary import (  # noqa: E402
    ANCHOR_BIT,
    FULL_MASK,
    N_PLAYERS,
    NEIGHBORHOODS,
    mask_to_coalition,
)

FLOPS_TOTAL_FORMULA = (
    "stem_epochs*(stem_train_epoch + val_pass) + "
    "n_game_evaluations*val_pass + "
    "epochs_continued*(mask_train_epoch + val_pass) + val_pass + test_pass"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--passes", type=int, default=128)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "anchored3_exact"))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"anchored3_{args.dataset}_seed{args.seed}.jsonl"
    if cell_complete(final_path):
        print(f"skip: {final_path} already complete")
        return
    run_dir = Path(
        args.run_dir or out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"patches: {PATCHES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neighborhoods = list(NEIGHBORHOODS)

    # ---- 1. stem ------------------------------------------------------------
    stem_cfg = drv.compose_config(
        dataset=args.dataset,
        seed=args.seed,
        neighborhoods=neighborhoods,
        output_dir=str(run_dir / "stem"),
        model=f"{drv.model_domain(args.dataset)}/{args.model}",
    )
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    # ---- 2. production sampled game -----------------------------------------
    # The game is journaled per distinct mask so a wall-clock kill (90-min
    # per-attempt cap) resumes instead of recomputing: on restart the stem is
    # retrained with the identical protocol/seed and journaled values are
    # replayed as-is (measured once, never re-estimated). game_seconds is
    # reconstructed from the journal's per-eval seconds, so wall accounting
    # stays honest across attempts. Disclosed per row when a resume happened.
    inner = model.backbone.backbone
    wrapper = CoalitionMaskedBackbone(inner)
    val_loader = pipe.datamodule.val_dataloader()
    m_val = {}

    journal_path = run_dir / "game_journal.jsonl"
    journal: dict[int, dict] = {}
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated tail from a mid-write kill; drop it
            journal[int(e["mask"])] = e
    n_replayed = len(journal)
    journal_fh = open(journal_path, "a")

    def evaluate(mask: int) -> float:
        if mask in journal:
            e = journal[mask]
            m_val["n"] = int(e["n"])
            return float(e["v"])
        t_eval = time.perf_counter()
        with wrapper.coalition(mask):
            acc, n = drv.accuracy_over_loader(
                model, val_loader, "Validation", device
            )
        m_val["n"] = n
        entry = {"mask": int(mask), "v": float(acc), "n": int(n),
                 "seconds": time.perf_counter() - t_eval}
        journal_fh.write(json.dumps(entry) + "\n")
        journal_fh.flush()
        journal[int(mask)] = entry
        return acc

    game = CachedGame(n_players=N_PLAYERS, evaluate=evaluate)
    t0 = time.perf_counter()
    v_full = game(FULL_MASK)
    attribution = sampled_shapley(
        game, N_PLAYERS, passes=args.passes, seed=args.seed, antithetic=True
    )
    journal_fh.close()
    # honest wall: every journaled eval's measured seconds (all attempts),
    # plus this attempt's non-eval overhead in the game loop
    replayed_seconds = sum(
        float(journal[m]["seconds"]) for m in journal
    )
    game_seconds = (time.perf_counter() - t0 if n_replayed == 0
                    else replayed_seconds)

    # ---- 3. anchored top-k ---------------------------------------------------
    picked = anchored_top_k_mask(attribution.phi, args.k,
                                 anchor_bit=ANCHOR_BIT)
    print(
        f"anchored k={args.k} pick: mask={picked} "
        f"members={list(mask_to_coalition(picked))} "
        f"phi={[round(float(p), 5) for p in attribution.phi]} "
        f"evals={game.calls}"
    )

    # ---- 4. warm prune + continuation + test --------------------------------
    inner.aggregate_inter_nbhd = wrapper._original_aggregate  # unhook game
    stem_snapshot = copy.deepcopy(model)  # full architecture, for profiling
    kept = prune_backbone_(inner, picked)
    model.to(device)

    continue_cfg = drv.compose_config(
        dataset=args.dataset,
        seed=args.seed,
        neighborhoods=neighborhoods,
        output_dir=str(run_dir / "continue"),
        model=f"{drv.model_domain(args.dataset)}/{args.model}",
    )
    t0 = time.perf_counter()
    _, ckpt_cb, _, epochs = drv.fit_continuation(continue_cfg, pipe)
    continue_seconds = time.perf_counter() - t0
    report = drv.test_best_checkpoint(continue_cfg, pipe, ckpt_cb, device)
    val_accuracy = float(report["val_metrics"]["val/accuracy"])
    test_accuracy = float(report["test_metrics"]["test/accuracy"])

    # ---- 5. measured FLOPs (after ALL training) ------------------------------
    train_loader = pipe.datamodule.train_dataloader()
    test_loader = pipe.datamodule.test_dataloader()
    stem_snapshot.to(device)
    stem_train_prof = profile_train_epoch(stem_snapshot, train_loader, device)
    val_pass_prof = profile_eval_pass(stem_snapshot, val_loader, device)
    test_pass_prof = profile_eval_pass(model, test_loader, device, "Test")
    mask_train_prof = profile_train_epoch(model, train_loader, device)

    vp, tp = val_pass_prof["flops"], test_pass_prof["flops"]
    flops_total = (
        args.stem_epochs * (stem_train_prof["flops"] + vp)
        + game.calls * vp
        + epochs * (mask_train_prof["flops"] + vp) + vp + tp
    )
    flops = {
        "conventions": GCCN_COUNTER_NOTE,
        "total_formula": FLOPS_TOTAL_FORMULA,
        "note": "lifting preprocessing excluded, matching #14's per-run "
                "accounting; eval passes profiled on the full stem "
                "architecture, mask train-epoch on the pruned model",
        "stem_train_epoch": stem_train_prof,
        "val_pass": val_pass_prof,
        "test_pass": test_pass_prof,
        "mask_train_epoch": mask_train_prof,
        "n_epochs_continued": int(epochs),
        "total_flops": int(flops_total),
    }
    print(f"flops: total {flops_total:.3e}")

    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selector": "anchored_k3",
        "meta": "production sampled game: 128 antithetic permutation "
                "passes, unique-evaluation accounting (CachedGame.calls); "
                "anchor = up_adjacency-0 (bit 0); measured FLOPs inline",
        "phi": [float(p) for p in attribution.phi],
        "mask": int(picked),
        "k": int(args.k),
        "members": kept,
        "v_full": float(v_full),
        "m_val": int(m_val["n"]),
        "monitored_metric": str(stem_cfg.dataset.parameters.monitor_metric),
        "floor_metric": "accuracy",
        "floor_note": "game values and the binomial-se floor use plain val accuracy; checkpointing/early-stop monitor the dataset's standard metric (conservative stand-in, disclosed, when they differ)",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "stem_seconds": float(stem_seconds),
        "game_seconds": float(game_seconds),
        "continue_seconds": float(continue_seconds),
        "n_game_evaluations": int(game.calls),
        "n_passes": int(args.passes),
        "antithetic": True,
        "game_resumed": n_replayed > 0,
        "n_game_evals_replayed": int(n_replayed),
        "game_resume_note": (
            "cell resumed after a wall-clock kill: stem retrained with the "
            "identical protocol/seed; journaled game values replayed as "
            "measured (never re-estimated); game_seconds = sum of per-eval "
            "measured seconds across attempts" if n_replayed else "-"),
        "n_epochs_trained_in_continuation": int(epochs),
        "selected_epoch_in_continuation": int(report["selected_epoch"]),
        "flops": flops,
        "n_parameters_backbone": drv.count_parameters(inner),
        "n_parameters_total": drv.count_parameters(model),
        "stem_epochs": int(args.stem_epochs),
        "model": args.model,
        "orientation": "src_to_dst",
        "patches": PATCHES,
    }
    out_path = out_dir / f"anchored3_{args.dataset}_seed{args.seed}.jsonl"
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}")
    print(
        f"k={args.k} mask={picked} val={val_accuracy:.4f} "
        f"test={test_accuracy:.4f} "
        f"wall={stem_seconds + game_seconds + continue_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
