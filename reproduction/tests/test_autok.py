"""Automatic-k rules and the submodularity census on known games."""

import numpy as np

from toposhap.selection import (
    cost_regularized,
    greedy_with_noise_stop,
    smallest_sufficient_coalition,
    submodularity_census,
    threshold_at_zero,
)
from toposhap.shapley import TabulatedGame, shapley_values


def additive_game(weights):
    n = len(weights)
    values = {
        m: float(sum(w for i, w in enumerate(weights) if m >> i & 1))
        for m in range(1 << n)
    }
    return TabulatedGame(n_players=n, values=values)


def coverage_game(sets, n_players):
    """Set-cover value function — canonically submodular."""
    values = {}
    for m in range(1 << n_players):
        covered = set()
        for i in range(n_players):
            if m >> i & 1:
                covered |= sets[i]
        values[m] = float(len(covered))
    return TabulatedGame(n_players=n_players, values=values)


def test_threshold_at_zero_on_additive_game():
    game = additive_game([0.5, -0.1, 0.3, 0.0, -0.2])
    phi = shapley_values(game, 5)  # = weights exactly, for additive games
    assert threshold_at_zero(phi) == 0b00101


def test_cost_regularized_prunes_cheap_players():
    game = additive_game([0.5, 0.05, 0.3, 0.02])
    phi = shapley_values(game, 4)
    assert cost_regularized(phi, cost_per_player=0.1) == 0b0101
    assert cost_regularized(phi, cost_per_player=0.01) == 0b1111


def test_greedy_noise_stop_ignores_subnoise_gains():
    game = additive_game([0.5, 0.3, 0.008, 0.006])
    picked = greedy_with_noise_stop(game, 4, noise_sd=0.014)
    # 0.008/0.006 gains are below the retraining-noise floor
    assert picked == 0b0011


def test_smallest_sufficient_coalition():
    game = additive_game([0.6, 0.3, 0.05, 0.05])
    phi = shapley_values(game, 4)
    picked = smallest_sufficient_coalition(game, 4, tolerance=0.15, phi=phi)
    assert picked == 0b0011  # 0.9 >= 1.0 - 0.15, and no single player suffices
    tight = smallest_sufficient_coalition(game, 4, tolerance=0.0, phi=phi)
    assert tight == 0b1111


def test_census_clean_on_submodular_game():
    sets = [{1, 2, 3}, {3, 4}, {4, 5, 6}, {1, 6}]
    game = coverage_game(sets, 4)
    census = submodularity_census(game)
    assert census.violations == 0
    assert census.triples_checked > 0


def test_census_catches_supermodular_pair():
    # complementarity: payoff only when BOTH 0 and 1 present
    n = 3
    values = {m: float(m & 0b11 == 0b11) for m in range(1 << n)}
    game = TabulatedGame(n_players=n, values=values)
    census = submodularity_census(game)
    assert census.violations > 0
    assert census.worst_violation < 0


def test_census_matches_known_rate_small():
    # additive games are modular: zero violations, exhaustively checked
    game = additive_game([0.1, 0.2, 0.3])
    census = submodularity_census(game)
    assert census.violation_rate == 0.0


def test_nemhauser_guarantee_on_submodular_game():
    """On a submodular game, greedy-k must reach (1 - 1/e) of the best-k —
    the guarantee the paper invokes iff the census is clean."""
    from toposhap.selection import greedy_mask

    sets = [{1, 2}, {2, 3}, {4}, {5, 6, 7}, {1, 7}]
    game = coverage_game(sets, 5)
    for k in (1, 2, 3):
        greedy_val = game(greedy_mask(game, 5, k=k))
        best_val = game.argmax_at_size(k)[1]
        assert greedy_val >= (1 - 1 / np.e) * best_val - 1e-12
