"""Automatic-k rules: choose the coalition size, not just the coalition.

Four principled rules, scored against exact landscape truth (the #14 rule
battery). Each returns a coalition mask; k falls out of the rule rather than
being a practitioner input. The paper's "automatic k" subsection reports how
each rule's picks score against the exhaustive landscape.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from toposhap.selection.selectors import greedy_mask, top_k_mask


def threshold_at_zero(phi: np.ndarray) -> int:
    """Keep every player with a strictly positive Shapley value."""
    mask = 0
    for i, p in enumerate(phi):
        if p > 0:
            mask |= 1 << i
    return mask


def cost_regularized(phi: np.ndarray, cost_per_player: float) -> int:
    """Keep players whose Shapley value exceeds a per-player cost.

    Equivalent to maximising sum(phi_i) - c*k over top-k prefixes when phi is
    sorted, but stated per-player so ties and negatives behave sensibly.
    """
    mask = 0
    for i, p in enumerate(phi):
        if p > cost_per_player:
            mask |= 1 << i
    return mask


def greedy_with_noise_stop(
    game: Callable[[int], float],
    n_players: int,
    noise_sd: float,
    start_mask: int = 0,
    max_k: int | None = None,
) -> int:
    """Greedy on the game; stop when the marginal gain drops below the noise
    floor.

    ``noise_sd`` should be the measured fixed-subset retraining noise for the
    dataset (0.003-0.014 sd on this benchmark suite, from #12's error-bar
    decomposition) — the rule stops once gains are indistinguishable from
    retraining noise.
    """
    limit = max_k if max_k is not None else n_players
    mask, trace = greedy_mask(
        game, n_players, k=limit, start_mask=start_mask, return_trace=True
    )
    kept = start_mask
    for bit, gain in trace:
        if gain < noise_sd:
            break
        kept |= 1 << bit
    return kept


def smallest_sufficient_coalition(
    game: Callable[[int], float],
    n_players: int,
    tolerance: float,
    phi: np.ndarray | None = None,
) -> int:
    """Smallest top-k prefix whose value is within ``tolerance`` of the grand
    coalition's value.

    Candidate prefixes are ordered by Shapley value when ``phi`` is given
    (cheap: n game evaluations), otherwise by greedy trace. Falls back to the
    grand coalition if nothing smaller suffices.
    """
    full = (1 << n_players) - 1
    target = game(full) - tolerance
    if phi is not None:
        for k in range(1, n_players + 1):
            mask = top_k_mask(phi, k)
            if game(mask) >= target:
                return mask
        return full
    _, trace = greedy_mask(game, n_players, k=n_players, return_trace=True)
    mask = 0
    for bit, _ in trace:
        mask |= 1 << bit
        if game(mask) >= target:
            return mask
    return full
