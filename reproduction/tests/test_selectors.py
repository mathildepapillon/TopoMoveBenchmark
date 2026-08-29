"""Selector behaviour, including the substitute-commitment failure mode."""

import numpy as np

from toposhap.selection import (
    anchored_top_k_mask,
    greedy_mask,
    top_k_mask,
)
from toposhap.shapley import TabulatedGame, shapley_values


def substitute_game(n=4):
    """Players 0 and 1 are perfect substitutes for a big payoff; players 2,3
    add small independent value. The exact Shapley values split the payoff
    between the substitutes, so plain top-2 takes BOTH substitutes and leaves
    real marginal value on the table — the wrong-rule-under-redundancy result
    (Fryer et al.) our data reproduces."""
    values = {}
    for m in range(1 << n):
        v = 1.0 * bool(m & 0b11)  # substitutes: either one unlocks 1.0
        v += 0.3 * bool(m & 0b100)  # player 2
        v += 0.2 * bool(m & 0b1000)  # player 3
        values[m] = v
    return TabulatedGame(n_players=n, values=values)


def test_plain_topk_commits_to_both_substitutes():
    game = substitute_game()
    phi = shapley_values(game, 4)
    # substitutes share the 1.0 payoff: phi_0 = phi_1 = 0.5 > 0.3 > 0.2
    picked = top_k_mask(phi, k=2)
    assert picked == 0b0011  # both substitutes — redundant pick
    assert game(picked) < game(0b0101)  # strictly worse than {0, 2}


def test_greedy_resolves_redundancy():
    game = substitute_game()
    picked = greedy_mask(game, 4, k=2)
    # after one substitute, the other's marginal collapses to 0
    assert picked in (0b0101, 0b0110)  # one substitute + player 2
    assert game(picked) == 1.3


def test_greedy_trace_gains_are_diminishing_here():
    game = substitute_game()
    _, trace = greedy_mask(game, 4, k=4, return_trace=True)
    gains = [g for _, g in trace]
    assert gains == sorted(gains, reverse=True)


def test_anchored_always_keeps_bit0():
    phi = np.array([-5.0, 1.0, 2.0, 3.0])  # anchor looks worthless
    for k in range(1, 5):
        assert anchored_top_k_mask(phi, k) & 1


def test_anchored_equals_plain_when_anchor_wins():
    phi = np.array([9.0, 1.0, 2.0, 3.0])
    for k in range(1, 5):
        assert anchored_top_k_mask(phi, k) == top_k_mask(phi, k)


def test_topk_deterministic_ties():
    phi = np.array([1.0, 1.0, 1.0, 1.0])
    assert top_k_mask(phi, 2) == 0b0011  # lower bits win ties


def test_greedy_with_start_mask():
    game = substitute_game()
    picked = greedy_mask(game, 4, k=2, start_mask=0b1)  # anchored greedy
    assert picked & 1
    assert picked == 0b0101  # anchor + player 2, substitute 1 skipped
