"""Exactness of the Shapley engine vs an independent brute force.

The brute force lives here, not in the library, so implementation and oracle
cannot drift together. The bar matches experiment #1: max error at machine
precision (the original recorded 2.254e-14 vs brute force).
"""

import itertools
import math

import numpy as np
import pytest

from toposhap.shapley import (
    TabulatedGame,
    efficiency_gap,
    shapley_interaction,
    shapley_values,
)

EXACTNESS_BAR = 5e-13  # same order as the original 2.254e-14 record


def brute_force_shapley(values: dict[int, float], n: int) -> np.ndarray:
    """Average marginal contribution over all n! permutations. O(n! n)."""
    phi = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for perm in perms:
        mask = 0
        for player in perm:
            phi[player] += values[mask | 1 << player] - values[mask]
            mask |= 1 << player
    return phi / len(perms)


def random_game(n: int, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    values = {m: float(rng.normal()) for m in range(1 << n)}
    values[0] = 0.0
    return values


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_matches_brute_force(n, seed):
    values = random_game(n, seed)
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    oracle = brute_force_shapley(values, n)
    assert np.max(np.abs(phi - oracle)) < EXACTNESS_BAR


@pytest.mark.parametrize("n", [3, 6, 11])
def test_efficiency_axiom(n):
    values = random_game(n, seed=42)
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    assert efficiency_gap(phi, game, n) < EXACTNESS_BAR


def test_null_player_gets_zero():
    # player 2 never changes the value
    n = 4
    rng = np.random.default_rng(7)
    base = {m: float(rng.normal()) for m in range(1 << (n - 1))}

    def squeeze(mask):  # value ignores bit 2
        low = mask & 0b0011
        high = (mask >> 3) << 2
        return base[low | high]

    values = {m: squeeze(m) for m in range(1 << n)}
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    assert abs(phi[2]) < EXACTNESS_BAR


def test_symmetric_players_get_equal_values():
    # v = number of {0,1} present (players 0,1 interchangeable), plus noise on 2,3
    n = 4
    rng = np.random.default_rng(3)
    extra = {m: float(rng.normal()) for m in range(1 << 2)}
    values = {
        m: bin(m & 0b11).count("1") + extra[m >> 2] for m in range(1 << n)
    }
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    assert abs(phi[0] - phi[1]) < EXACTNESS_BAR


def test_additive_game_has_zero_interactions():
    n = 5
    rng = np.random.default_rng(11)
    w = rng.normal(size=n)
    values = {
        m: float(sum(w[i] for i in range(n) if m >> i & 1))
        for m in range(1 << n)
    }
    game = TabulatedGame(n_players=n, values=values)
    inter = shapley_interaction(game, n, order=2)
    assert max(abs(v) for v in inter.values()) < EXACTNESS_BAR


def test_pure_pair_interaction_recovered():
    # v(S) = 1 iff both 0 and 1 in S: I({0,1}) must be exactly 1,
    # and interactions not involving the pair must be 0.
    n = 4
    values = {m: float(m & 0b11 == 0b11) for m in range(1 << n)}
    game = TabulatedGame(n_players=n, values=values)
    inter = shapley_interaction(game, n, order=2)
    assert abs(inter[frozenset({0, 1})] - 1.0) < EXACTNESS_BAR
    assert abs(inter[frozenset({2, 3})]) < EXACTNESS_BAR


def test_interaction_weights_sum_property():
    # sanity on the Grabisch-Roubens normalisation: for n=2, I({0,1}) equals
    # v(12) - v(1) - v(2) + v(0) exactly.
    values = {0: 0.0, 1: 1.0, 2: 2.0, 3: 5.0}
    game = TabulatedGame(n_players=2, values=values)
    inter = shapley_interaction(game, 2, order=2)
    expected = values[3] - values[1] - values[2] + values[0]
    assert math.isclose(inter[frozenset({0, 1})], expected, abs_tol=1e-14)
