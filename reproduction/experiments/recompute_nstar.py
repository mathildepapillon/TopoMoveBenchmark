#!/usr/bin/env python
"""Recompute N* and regrets from frozen data (post-marker-arm).

N* (an EVALUATION metric, not a practitioner input): the number of standard
training runs a validation-guided sweep needs before it matches the recipe's
pick quality — computed per dataset by comparing the recipe's test accuracy
against the best-of-N distribution of sweep runs. The #12 verdict was 6/7
datasets with N* >= 2 (Fluoride 1.2 and NCI1 1.5 the stated exceptions,
diagnosed as adjacency-drop and expected to be repaired by the anchored arm —
rerun this after markers land).

Regret: exact-game pick value minus recipe pick value on the same landscape
(<= +0.024 at 128 passes in #12).

    python experiments/recompute_nstar.py --selector anchored
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def best_of_n_curve(values: np.ndarray, rng, draws: int = 10_000):
    """E[max of n uniform draws] for n = 1..len(values), bootstrapped."""
    out = []
    for n in range(1, len(values) + 1):
        picks = rng.choice(values, size=(draws, n), replace=True).max(axis=1)
        out.append(float(picks.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selector", default="anchored",
                    choices=["anchored", "plain"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from toposhap.io import (
        load_landscape_as_game,
        load_marker_arm14,
        load_pooled9,
    )

    df = load_marker_arm14()
    pooled = load_pooled9()
    rng = np.random.default_rng(args.seed)

    recipe = df[df.selector == args.selector]
    print(f"selector = {args.selector}\n")
    for ds, grp in recipe.groupby("dataset"):
        recipe_acc = float(grp.test_accuracy.mean())

        try:
            game = load_landscape_as_game(pooled, ds)
        except KeyError:
            print(f"{ds:>14}: no landscape — regret/N* skipped")
            continue

        # sweep pool = the landscape's mask values (menu-like candidates)
        sweep_values = np.array(list(game.values.values()))
        curve = best_of_n_curve(sweep_values, rng)
        nstar = next(
            (n + 1 for n, v in enumerate(curve) if v >= recipe_acc),
            float("inf"),
        )

        # regret vs exact-game argmax at matching k values
        regrets = []
        for k, kgrp in grp.groupby("k"):
            try:
                _, exact_best = game.argmax_at_size(int(k))
            except KeyError:
                continue
            regrets.append(exact_best - float(kgrp.test_accuracy.mean()))
        regret = max(regrets) if regrets else float("nan")

        print(
            f"{ds:>14}: recipe {recipe_acc:.4f} | N* = {nstar} | "
            f"max regret vs exact-game pick = {regret:+.4f}"
        )


if __name__ == "__main__":
    main()
