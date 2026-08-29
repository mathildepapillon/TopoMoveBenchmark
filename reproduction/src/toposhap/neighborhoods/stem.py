"""The one-run recipe: stem -> sampled attribution -> prune -> continue.

Replaces architecture sweeps with one guided run at ~1.2-3.6 measured
training-run equivalents (experiment #12): train an all-11-neighborhood
ensemble stem for a few epochs, play the neighborhood game with the sampled
Shapley estimator (128 passes), prune to a coalition, continue training warm.
Picks match exact-game picks at ~1/10 cost with regret <= +0.024.

This module is deliberately trainer-agnostic: TopoBench's hydra/lightning
machinery differs per dataset, so the recipe consumes a :class:`RecipeHooks`
implementation (experiments/run_recipe.py wires one up for TopoBench). That
keeps the scientific logic — game construction, scoring convention, selector
choice, honest cost bookkeeping — in one tested place.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from toposhap.shapley.games import CachedGame
from toposhap.shapley.sampled import SampledAttribution, sampled_shapley
from toposhap.selection.selectors import anchored_top_k_mask, top_k_mask
from toposhap.vocabulary import ANCHOR_BIT, FULL_MASK, N_PLAYERS


class RecipeHooks(Protocol):
    """What the recipe needs from a concrete training stack."""

    def train_stem(self, epochs: int, seed: int) -> Any:
        """Train the all-neighborhood stem; returns an opaque model handle."""
        ...

    def val_accuracy_under_coalition(self, model: Any, mask: int) -> float:
        """Validation accuracy with the model masked to the coalition.

        Accuracy scoring, not logit scoring: accuracy-scored games beat
        logit-scored games for neighborhood attribution (experiment #8).
        """
        ...

    def prune_and_continue(self, model: Any, mask: int, seed: int) -> Any:
        """Prune to the coalition, continue training to completion (warm)."""
        ...

    def test_accuracy(self, model: Any) -> float:
        ...


@dataclass
class RecipeResult:
    seed: int
    selector: str
    k: int
    picked_mask: int
    attribution: SampledAttribution
    test_accuracy: float
    stem_seconds: float
    game_seconds: float
    continue_seconds: float
    game_evaluations: int
    extras: dict = field(default_factory=dict)


def run_recipe(
    hooks: RecipeHooks,
    seed: int,
    k: int,
    stem_epochs: int = 5,
    passes: int = 128,
    selector: str = "anchored",
    custom_selector: Callable[[np.ndarray], int] | None = None,
) -> RecipeResult:
    """One guided run. ``selector``: 'anchored' (default, always keeps
    up_adjacency-0), 'plain' (top-k, the honest baseline), or 'custom'."""
    t0 = time.perf_counter()
    stem = hooks.train_stem(epochs=stem_epochs, seed=seed)
    t1 = time.perf_counter()

    game = CachedGame(
        n_players=N_PLAYERS,
        evaluate=lambda mask: hooks.val_accuracy_under_coalition(stem, mask),
    )
    attribution = sampled_shapley(game, N_PLAYERS, passes=passes, seed=seed)
    t2 = time.perf_counter()

    if selector == "anchored":
        picked = anchored_top_k_mask(attribution.phi, k, anchor_bit=ANCHOR_BIT)
    elif selector == "plain":
        picked = top_k_mask(attribution.phi, k)
    elif selector == "custom":
        if custom_selector is None:
            raise ValueError("selector='custom' requires custom_selector")
        picked = custom_selector(attribution.phi)
    else:
        raise ValueError(f"unknown selector {selector!r}")
    if not 0 < picked <= FULL_MASK:
        raise ValueError(f"selector produced invalid mask {picked}")

    final = hooks.prune_and_continue(stem, picked, seed=seed)
    t3 = time.perf_counter()

    return RecipeResult(
        seed=seed,
        selector=selector,
        k=k,
        picked_mask=picked,
        attribution=attribution,
        test_accuracy=hooks.test_accuracy(final),
        stem_seconds=t1 - t0,
        game_seconds=t2 - t1,
        continue_seconds=t3 - t2,
        game_evaluations=game.calls,
    )
