"""Exactness of the Shapley engine vs an independent brute force.

The brute force lives here, not in the library, so implementation and
oracle cannot drift together. The bar is machine precision.
"""

import itertools
import math

import numpy as np
import pytest

from topobench.explain import (
    TabulatedGame,
    efficiency_gap,
    resample_pick_stability,
    sampled_shapley,
    shapley_interaction,
    shapley_values,
)

EXACTNESS_BAR = 5e-13


def brute_force_shapley(values, n):
    """Average marginal contribution over all n! permutations. O(n! n).

    Parameters
    ----------
    values : dict[int, float]
        Mapping from coalition bitmask to game value.
    n : int
        Number of players.

    Returns
    -------
    np.ndarray
        Shapley value per player.
    """
    phi = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for perm in perms:
        mask = 0
        for player in perm:
            phi[player] += values[mask | 1 << player] - values[mask]
            mask |= 1 << player
    return phi / len(perms)


def random_game(n, seed):
    """Build a random tabulated game with v(empty) = 0.

    Parameters
    ----------
    n : int
        Number of players.
    seed : int
        Seed for the random values.

    Returns
    -------
    dict[int, float]
        Mapping from coalition bitmask to game value.
    """
    rng = np.random.default_rng(seed)
    values = {m: float(rng.normal()) for m in range(1 << n)}
    values[0] = 0.0
    return values


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_matches_brute_force(n, seed):
    """Exact values must match the permutation brute force to precision."""
    values = random_game(n, seed)
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    oracle = brute_force_shapley(values, n)
    assert np.max(np.abs(phi - oracle)) < EXACTNESS_BAR


@pytest.mark.parametrize("n", [3, 6, 11])
def test_efficiency_axiom(n):
    """Exact values must satisfy sum(phi) == v(N) - v(empty)."""
    values = random_game(n, seed=42)
    game = TabulatedGame(n_players=n, values=values)
    phi = shapley_values(game, n)
    assert efficiency_gap(phi, game, n) < EXACTNESS_BAR


def test_null_player_gets_zero():
    """A player that never changes the value must get phi == 0."""
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
    """Interchangeable players must receive equal Shapley values."""
    # v = number of {0,1} present (players 0,1 interchangeable),
    # plus noise depending on players 2,3 only
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
    """Additive games must have identically zero pairwise interactions."""
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
    """A pure AND game must yield I({0,1}) == 1 and 0 elsewhere."""
    # v(S) = 1 iff both 0 and 1 in S
    n = 4
    values = {m: float(m & 0b11 == 0b11) for m in range(1 << n)}
    game = TabulatedGame(n_players=n, values=values)
    inter = shapley_interaction(game, n, order=2)
    assert abs(inter[frozenset({0, 1})] - 1.0) < EXACTNESS_BAR
    assert abs(inter[frozenset({2, 3})]) < EXACTNESS_BAR


def test_interaction_weights_sum_property():
    """For n=2, I({0,1}) equals v(12) - v(1) - v(2) + v(0) exactly."""
    values = {0: 0.0, 1: 1.0, 2: 2.0, 3: 5.0}
    game = TabulatedGame(n_players=2, values=values)
    inter = shapley_interaction(game, 2, order=2)
    expected = values[3] - values[1] - values[2] + values[0]
    assert math.isclose(inter[frozenset({0, 1})], expected, abs_tol=1e-14)


def test_sampled_converges_to_exact():
    """Sampled estimates must land within a few stderr of exact values."""
    n = 11
    game = TabulatedGame(n_players=n, values=random_game(n, seed=5))
    exact = shapley_values(game, n)
    att = sampled_shapley(game, n, passes=512, seed=0)
    # within 5 standard errors elementwise (generous, deterministic seed)
    assert np.all(np.abs(att.phi - exact) < 5 * att.stderr + 1e-9)


def test_sampled_efficiency_holds_per_permutation():
    """Permutation sampling satisfies efficiency exactly for every pass."""
    n = 8
    game = TabulatedGame(n_players=n, values=random_game(n, seed=9))
    att = sampled_shapley(game, n, passes=16, seed=1)
    full = (1 << n) - 1
    assert abs(att.phi.sum() - (game(full) - game(0))) < 1e-10


def test_sampled_evaluation_accounting_bounded():
    """Distinct evaluations must be bounded by n * passes + 1."""
    n = 7
    game = TabulatedGame(n_players=n, values=random_game(n, seed=2))
    att = sampled_shapley(game, n, passes=64, seed=3)
    assert att.evaluations <= 64 * n + 1
    assert att.passes == 64


def test_sampled_reproducible_given_seed():
    """Two runs with the same seed must agree bit for bit."""
    n = 6
    game = TabulatedGame(n_players=n, values=random_game(n, seed=4))
    a = sampled_shapley(game, n, passes=32, seed=7)
    b = sampled_shapley(game, n, passes=32, seed=7)
    assert np.array_equal(a.phi, b.phi)


def test_pick_stability_flags_substitutes():
    """Substitutes yield unstable picks; a dominant player, stable ones."""
    n = 4

    # v = 1 if player 0 OR player 1 present (perfect substitutes),
    # + 0.5 if player 2 present. k=1 picks straddle {0} vs {1}.
    values = {
        m: float(bool(m & 0b11)) + 0.5 * bool(m & 0b100)
        for m in range(1 << n)
    }
    sub_game = TabulatedGame(n_players=n, values=values)
    # antithetic=False: with pairing, exact substitutes tie every run and
    # the deterministic tie-break would mask the instability.
    unstable = resample_pick_stability(
        sub_game, n, passes=8, k=1, n_resamples=40, seed=0, antithetic=False
    )

    # dominant player 0: picks should be unanimous at the same budget
    dom_values = {
        m: 2.0 * bool(m & 1) + 0.1 * bin(m).count("1")
        for m in range(1 << n)
    }
    dom_game = TabulatedGame(n_players=n, values=dom_values)
    stable = resample_pick_stability(
        dom_game, n, passes=8, k=1, n_resamples=40, seed=0
    )

    assert stable["modal_agreement"] == 1.0
    assert unstable["modal_agreement"] < 1.0
