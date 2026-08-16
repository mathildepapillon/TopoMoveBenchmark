"""Selector behavior, automatic-size rules, and the submodularity census."""

import numpy as np

from topobench.explain import (
    TabulatedGame,
    anchored_top_k_mask,
    cost_regularized,
    greedy_mask,
    greedy_with_noise_stop,
    shapley_values,
    smallest_sufficient_coalition,
    submodularity_census,
    threshold_at_zero,
    top_k_mask,
)


def substitute_game(n=4):
    """Build a game with two perfect substitutes and two small players.

    Players 0 and 1 are perfect substitutes for a big payoff; players 2, 3
    add small independent value. Exact Shapley values split the payoff
    between the substitutes, so plain top-2 takes BOTH substitutes and
    leaves real marginal value on the table — the wrong-rule-under-
    redundancy result of Fryer, Strumke & Nguyen (2021).

    Parameters
    ----------
    n : int
        Number of players (the first two are the substitutes).

    Returns
    -------
    TabulatedGame
        The substitute game.
    """
    values = {}
    for m in range(1 << n):
        v = 1.0 * bool(m & 0b11)  # substitutes: either one unlocks 1.0
        v += 0.3 * bool(m & 0b100)  # player 2
        v += 0.2 * bool(m & 0b1000)  # player 3
        values[m] = v
    return TabulatedGame(n_players=n, values=values)


def additive_game(weights):
    """Build an additive (modular) game from per-player weights.

    Parameters
    ----------
    weights : list[float]
        Weight of each player.

    Returns
    -------
    TabulatedGame
        The additive game.
    """
    n = len(weights)
    values = {
        m: float(sum(w for i, w in enumerate(weights) if m >> i & 1))
        for m in range(1 << n)
    }
    return TabulatedGame(n_players=n, values=values)


def coverage_game(sets, n_players):
    """Build a set-cover game — canonically submodular.

    Parameters
    ----------
    sets : list[set]
        The set covered by each player.
    n_players : int
        Number of players.

    Returns
    -------
    TabulatedGame
        The coverage game.
    """
    values = {}
    for m in range(1 << n_players):
        covered = set()
        for i in range(n_players):
            if m >> i & 1:
                covered |= sets[i]
        values[m] = float(len(covered))
    return TabulatedGame(n_players=n_players, values=values)


def test_plain_topk_commits_to_both_substitutes():
    """Plain top-k must pick both substitutes — the redundant pick."""
    game = substitute_game()
    phi = shapley_values(game, 4)
    # substitutes share the 1.0 payoff: phi_0 = phi_1 = 0.5 > 0.3 > 0.2
    picked = top_k_mask(phi, k=2)
    assert picked == 0b0011  # both substitutes — redundant pick
    assert game(picked) < game(0b0101)  # strictly worse than {0, 2}


def test_greedy_resolves_redundancy():
    """Greedy must skip the second substitute once one is in."""
    game = substitute_game()
    picked = greedy_mask(game, 4, k=2)
    # after one substitute, the other's marginal collapses to 0
    assert picked in (0b0101, 0b0110)  # one substitute + player 2
    assert game(picked) == 1.3


def test_greedy_trace_gains_are_diminishing_here():
    """The greedy trace on this game must have non-increasing gains."""
    game = substitute_game()
    _, trace = greedy_mask(game, 4, k=4, return_trace=True)
    gains = [g for _, g in trace]
    assert gains == sorted(gains, reverse=True)


def test_anchored_always_keeps_anchor():
    """The anchor must be kept even when it looks worthless."""
    phi = np.array([-5.0, 1.0, 2.0, 3.0])  # anchor looks worthless
    for k in range(1, 5):
        assert anchored_top_k_mask(phi, k, anchor_bit=0) & 1

    phi2 = np.array([1.0, 2.0, -5.0, 3.0])  # anchor on another bit
    for k in range(1, 5):
        assert anchored_top_k_mask(phi2, k, anchor_bit=2) & 0b100


def test_anchored_equals_plain_when_anchor_wins():
    """Anchoring the top player must reduce to plain top-k."""
    phi = np.array([9.0, 1.0, 2.0, 3.0])
    for k in range(1, 5):
        assert anchored_top_k_mask(phi, k, anchor_bit=0) == top_k_mask(
            phi, k
        )


def test_topk_deterministic_ties():
    """Exact ties must break toward the lower bit index."""
    phi = np.array([1.0, 1.0, 1.0, 1.0])
    assert top_k_mask(phi, 2) == 0b0011  # lower bits win ties


def test_greedy_with_start_mask():
    """A seeded (anchored) greedy must keep the seed and stay greedy."""
    game = substitute_game()
    picked = greedy_mask(game, 4, k=2, start_mask=0b1)  # anchored greedy
    assert picked & 1
    assert picked == 0b0101  # anchor + player 2, substitute 1 skipped


def test_threshold_at_zero_on_additive_game():
    """Threshold-at-zero must keep exactly the positive-weight players."""
    game = additive_game([0.5, -0.1, 0.3, 0.0, -0.2])
    phi = shapley_values(game, 5)  # = weights exactly, for additive games
    assert threshold_at_zero(phi) == 0b00101


def test_cost_regularized_prunes_cheap_players():
    """Cost regularization must drop players below the per-player cost."""
    game = additive_game([0.5, 0.05, 0.3, 0.02])
    phi = shapley_values(game, 4)
    assert cost_regularized(phi, cost_per_player=0.1) == 0b0101
    assert cost_regularized(phi, cost_per_player=0.01) == 0b1111


def test_greedy_noise_stop_ignores_subnoise_gains():
    """Gains below the noise floor must not extend the coalition."""
    game = additive_game([0.5, 0.3, 0.008, 0.006])
    picked = greedy_with_noise_stop(game, 4, noise_sd=0.014)
    assert picked == 0b0011


def test_smallest_sufficient_coalition():
    """The smallest prefix within tolerance of the full value must win."""
    game = additive_game([0.6, 0.3, 0.05, 0.05])
    phi = shapley_values(game, 4)
    picked = smallest_sufficient_coalition(game, 4, tolerance=0.15, phi=phi)
    assert picked == 0b0011  # 0.9 >= 1.0 - 0.15; no single player suffices
    tight = smallest_sufficient_coalition(game, 4, tolerance=0.0, phi=phi)
    assert tight == 0b1111


def test_census_clean_on_submodular_game():
    """A coverage game must produce zero violations."""
    sets = [{1, 2, 3}, {3, 4}, {4, 5, 6}, {1, 6}]
    game = coverage_game(sets, 4)
    census = submodularity_census(game)
    assert census.violations == 0
    assert census.triples_checked > 0


def test_census_catches_supermodular_pair():
    """A complementarity (AND) game must produce violations."""
    # payoff only when BOTH 0 and 1 present
    n = 3
    values = {m: float(m & 0b11 == 0b11) for m in range(1 << n)}
    game = TabulatedGame(n_players=n, values=values)
    census = submodularity_census(game)
    assert census.violations > 0
    assert census.worst_violation < 0


def test_census_zero_rate_on_modular_game():
    """Additive games are modular: zero violations, checked exhaustively."""
    game = additive_game([0.1, 0.2, 0.3])
    census = submodularity_census(game)
    assert census.violation_rate == 0.0


def test_nemhauser_guarantee_on_submodular_game():
    """On a submodular game, greedy-k must reach (1 - 1/e) of best-k."""
    sets = [{1, 2}, {2, 3}, {4}, {5, 6, 7}, {1, 7}]
    game = coverage_game(sets, 5)
    for k in (1, 2, 3):
        greedy_val = game(greedy_mask(game, 5, k=k))
        best_val = game.argmax_at_size(k)[1]
        assert greedy_val >= (1 - 1 / np.e) * best_val - 1e-12
