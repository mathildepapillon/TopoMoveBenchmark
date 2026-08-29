"""Subset selectors: attribution -> coalition mask.

Three selectors, in ascending order of robustness to redundancy:

* :func:`top_k_mask` — plain top-k by Shapley value. Known to be the wrong
  rule under redundancy (Fryer, Strumke & Nguyen 2021); kept as the honest
  baseline our data independently reproduces.
* :func:`anchored_top_k_mask` — always keep the anchor (up_adjacency-0,
  bit 0), fill remaining slots by Shapley value. Best prior selector in
  experiment #9 (adjacency-anchored argmax, 5/11 datasets, +0.0301); the
  pre-registered Q3 fix in #14 for substitute-commitment.
* :func:`greedy_mask` — greedy maximisation on the game itself. Handles
  redundancy structurally (after one substitute is added the other's marginal
  collapses) and carries the Nemhauser (1-1/e) guarantee iff the game is
  submodular — which we measure (toposhap.selection.submodularity), never
  assume.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def top_k_mask(phi: np.ndarray, k: int) -> int:
    """Coalition of the k players with the largest Shapley values.

    Ties break toward the lower bit index (deterministic across runs).
    """
    order = np.lexsort((np.arange(len(phi)), -phi))
    mask = 0
    for i in order[:k]:
        mask |= 1 << int(i)
    return mask


def anchored_top_k_mask(phi: np.ndarray, k: int, anchor_bit: int = 0) -> int:
    """Top-k that always includes the anchor player.

    The anchor occupies one of the k slots; the rest fill by Shapley value.
    """
    if k < 1:
        raise ValueError("anchored selection needs k >= 1")
    mask = 1 << anchor_bit
    order = np.lexsort((np.arange(len(phi)), -phi))
    for i in order:
        if bin(mask).count("1") >= k:
            break
        if int(i) != anchor_bit:
            mask |= 1 << int(i)
    return mask


def greedy_mask(
    game: Callable[[int], float],
    n_players: int,
    k: int,
    start_mask: int = 0,
    return_trace: bool = False,
):
    """Greedy forward selection on the game, up to k players.

    At each step add the player with the largest marginal gain on the game
    value. ``start_mask`` seeds the coalition (pass ``1 << 0`` for an anchored
    greedy). With ``return_trace=True`` also returns the per-step (bit, gain)
    list, which the greedy-with-noise-stop automatic-k rule consumes.
    """
    mask = start_mask
    trace: list[tuple[int, float]] = []
    while bin(mask).count("1") < k:
        best_bit, best_gain = None, -np.inf
        base = game(mask)
        for i in range(n_players):
            bit = 1 << i
            if mask & bit:
                continue
            gain = game(mask | bit) - base
            if gain > best_gain:
                best_bit, best_gain = i, gain
        if best_bit is None:
            break
        mask |= 1 << best_bit
        trace.append((best_bit, float(best_gain)))
    if return_trace:
        return mask, trace
    return mask
