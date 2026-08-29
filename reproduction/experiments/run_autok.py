#!/usr/bin/env python
"""Auto-k greedy arm: stem -> masked val-accuracy game -> greedy with binomial
stop -> prune -> continue -> best-val-checkpoint test.

    python experiments/run_autok.py --dataset benzene --seed 42

Protocol (mirrors experiment #14, whose data this arm must be comparable to):

1. ``TOPOSHAP_INTERRANK_ORIENTATION=fixed`` + ``toposhap.patches.apply_all()``
   before building any model (#14 recorded orientation "src_to_dst").
2. TopoBench topotune model with the full 11-neighborhood vocabulary in
   canonical bit order (``toposhap.vocabulary.NEIGHBORHOODS``), dataset's
   standard TopoBench config; the model config that reproduces the #14
   parameter record is ``cell/topotune_nonlinear_exact`` (gate A: benzene
   mask 132 -> backbone 8704, total 10528 — exact). Stem: 5 epochs, seeded,
   no early stop.
3. Masked validation-accuracy game on the trained stem's inner TopoTune
   module (``model.backbone.backbone``) via CoalitionMaskedBackbone;
   v(S) = full-val-set accuracy under coalition S.
4. Greedy forward selection from the empty coalition; stop when the best
   marginal gain < sqrt(p(1-p)/m), p = v(full mask 2047), m = val-set size;
   always keep the first neighborhood (k >= 1).
5. Prune warm (``prune_backbone_``), continue training with the dataset's
   standard trainer settings (default max_epochs, val checkpointing,
   standard callbacks incl. early stopping), test the best-val checkpoint
   via TopoBench's normal test path.
6. Emit one JSON line to results/autok/autok_<dataset>_seed<seed>.jsonl.
"""

from __future__ import annotations

import argparse
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
from toposhap.neighborhoods import (  # noqa: E402
    CoalitionMaskedBackbone,
    prune_backbone_,
)
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.vocabulary import (  # noqa: E402
    FULL_MASK,
    N_PLAYERS,
    NEIGHBORHOODS,
)

DATASETS = [
    "benzene",
    "fluoride_carbonyl",
    "mutagenicity_gxai",
    "NCI1",
    "cocitation_cora",
    "cocitation_citeseer",
    "cocitation_pubmed",
    # Tier-1 coverage expansion (2026-08-17)
    "MUTAG",
    "NCI109",
    "alkane_carbonyl",
    # MANTRA addition (2026-08-17): native simplicial, orientability task
    "mantra_orientation",
]
SEEDS = [42, 43, 44, 45, 46]


def greedy_autok(game, threshold: float):
    """Greedy forward selection from empty with the binomial-SE stop.

    Returns (mask, trace, stopped, ties) where trace is the accepted
    [(bit, gain)] list and stopped is the first rejected candidate
    {"bit", "gain"} or None if selection ran through all players.
    """
    mask = 0
    base = game(mask)  # v(empty): baseline for the first marginal gain
    trace: list[tuple[int, float]] = []
    stopped = None
    ties: list[dict] = []
    while bin(mask).count("1") < N_PLAYERS:
        best_bit, best_value = None, -math.inf
        tie_bits: list[int] = []
        for i in range(N_PLAYERS):
            if mask >> i & 1:
                continue
            value = game(mask | (1 << i))
            if value > best_value:
                best_bit, best_value = i, value
                tie_bits = [i]
            elif value == best_value:
                tie_bits.append(i)
        gain = best_value - base
        if trace and gain < threshold:  # always keep the first pick (k >= 1)
            stopped = {"bit": best_bit, "gain": gain}
            break
        if len(tie_bits) > 1:
            ties.append({"step": len(trace), "bits": tie_bits,
                         "v": float(best_value)})
        mask |= 1 << best_bit
        trace.append((best_bit, float(gain)))
        base = best_value
    return mask, trace, stopped, ties


def backward_autok(game, threshold: float):
    """Backward elimination from the full coalition with the binomial stop.

    Start at mask 2047; at each step remove the member whose removal hurts
    least (largest v after removal); stop when the best candidate removal
    would drop v by MORE than the threshold (keep the current coalition).
    Floor: never below k = 1. Ties break toward the lowest bit index and are
    recorded.

    Returns (mask, trace, stopped, ties): trace is the executed
    [(bit_removed, v_after)] list; stopped is the first rejected removal
    {"bit", "v_after", "drop"} or None if elimination reached k = 1.
    """
    mask = FULL_MASK
    base = game(mask)
    trace: list[tuple[int, float]] = []
    stopped = None
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
        drop = base - best_value
        if drop > threshold:  # removal hurts more than the noise floor: keep S
            stopped = {"bit": best_bit, "v_after": float(best_value),
                       "drop": float(drop)}
            break
        if len(tie_bits) > 1:
            ties.append({"step": len(trace), "bits": tie_bits,
                         "v": float(best_value)})
        mask &= ~(1 << best_bit)
        trace.append((best_bit, float(best_value)))
        base = best_value
    return mask, trace, stopped, ties


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--direction", choices=["forward", "backward"],
                    default="forward",
                    help="forward = greedy from empty (selector "
                         "autok_greedy); backward = elimination from the "
                         "full coalition (selector autok_backward)")
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir", default=None,
                    help="default: results/autok (forward) or "
                         "results/autok_backward (backward)")
    ap.add_argument("--run-dir", default=None,
                    help="trainer output dir (default: <out-dir>/runs/...)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build dataset + preprocessor cache and exit "
                         "(CPU-safe; use on the login node to warm caches)")
    args = ap.parse_args()

    backward = args.direction == "backward"
    selector = "autok_backward" if backward else "autok_greedy"
    prefix = "autok_backward" if backward else "autok"
    out_dir = Path(
        args.out_dir
        or REPO / "results" / ("autok_backward" if backward else "autok")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        args.run_dir or out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"patches: {PATCHES}")
    neighborhoods = list(NEIGHBORHOODS)

    stem_cfg = drv.compose_config(
        dataset=args.dataset,
        seed=args.seed,
        neighborhoods=neighborhoods,
        output_dir=str(run_dir / "stem"),
        model=f"{drv.model_domain(args.dataset)}/{args.model}",
    )

    if args.prepare_only:
        pipe = drv.build_pipeline(stem_cfg, data_only=True)
        print(
            f"prepared {args.dataset}: preprocessing "
            f"{pipe.preprocessing_seconds:.1f}s"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1-2. build + stem ------------------------------------------------
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    # ---- 3. masked validation-accuracy game -------------------------------
    inner = model.backbone.backbone  # TBModel.backbone is a TuneWrapper
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

    # ---- 4. selection with binomial stop ----------------------------------
    select = backward_autok if backward else greedy_autok
    mask, trace, stopped, ties = select(game, threshold=binomial_se)
    game_seconds = time.perf_counter() - t0
    k = bin(mask).count("1")
    print(
        f"picked mask {mask} (k={k}, {selector}) trace={trace} "
        f"stopped={stopped} ties={ties} "
        f"v_full={v_full:.4f} se={binomial_se:.4f} m={m_val['n']} "
        f"evals={game.calls}"
    )

    # ---- 5. prune warm + continue with standard trainer settings ----------
    inner.aggregate_inter_nbhd = wrapper._original_aggregate  # unhook game
    kept = prune_backbone_(inner, mask)
    model.to(device)
    print(f"pruned to {kept}")

    continue_cfg = drv.compose_config(
        dataset=args.dataset,
        seed=args.seed,
        neighborhoods=neighborhoods,
        output_dir=str(run_dir / "continue"),
        model=f"{drv.model_domain(args.dataset)}/{args.model}",
    )
    trainer, ckpt_cb, continue_seconds, epochs_trained = drv.fit_continuation(
        continue_cfg, pipe
    )
    report = drv.test_best_checkpoint(continue_cfg, pipe, ckpt_cb, device)
    val_accuracy = float(report["val_metrics"]["val/accuracy"])
    test_accuracy = float(report["test_metrics"]["test/accuracy"])

    # ---- 6. one JSON line per (dataset, seed) ------------------------------
    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selector": selector,
        "trace": [[int(b), float(g)] for b, g in trace],
        "stopped": stopped,
        "ties": ties,
        "mask": int(mask),
        "k": int(k),
        "members": kept,
        "v_full": float(v_full),
        "v_empty": float(game(0)) if not backward else None,
        "binomial_se": float(binomial_se),
        "m_val": int(m_val["n"]),
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "stem_seconds": float(stem_seconds),
        "game_seconds": float(game_seconds),
        "continue_seconds": float(continue_seconds),
        "n_game_evaluations": int(game.calls),
        "n_epochs_trained_in_continuation": int(epochs_trained),
        "selected_epoch_in_continuation": int(report["selected_epoch"]),
        "n_parameters_backbone": drv.count_parameters(inner),
        "n_parameters_total": drv.count_parameters(model),
        "stem_epochs": int(args.stem_epochs),
        "model": args.model,
        "orientation": "src_to_dst",
        "patches": PATCHES,
        "best_model_path": report["best_model_path"],
    }
    out_path = out_dir / f"{prefix}_{args.dataset}_seed{args.seed}.jsonl"
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}")
    print(
        f"k={k} mask={mask} val={val_accuracy:.4f} test={test_accuracy:.4f} "
        f"wall={stem_seconds + game_seconds + continue_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
