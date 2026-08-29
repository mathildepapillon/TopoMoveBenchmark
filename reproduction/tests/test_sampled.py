"""Sampled-estimator convergence, cost accounting, and stability probing."""

import numpy as np

from toposhap.shapley import (
    TabulatedGame,
    resample_pick_stability,
    sampled_shapley,
    shapley_values,
)


def random_game(n, seed):
    rng = np.random.default_rng(seed)
    values = {m: float(rng.normal()) for m in range(1 << n)}
    values[0] = 0.0
    return TabulatedGame(n_players=n, values=values)


def test_converges_to_exact_on_11_players():
    n = 11
    game = random_game(n, seed=5)
    exact = shapley_values(game, n)
    att = sampled_shapley(game, n, passes=512, seed=0)
    # within 5 standard errors elementwise (generous, deterministic seed)
    assert np.all(np.abs(att.phi - exact) < 5 * att.stderr + 1e-9)


def test_efficiency_holds_per_permutation():
    # permutation sampling satisfies efficiency exactly for every pass, so the
    # mean does too: sum(phi) == v(N) - v(0) to float precision.
    n = 8
    game = random_game(n, seed=9)
    att = sampled_shapley(game, n, passes=16, seed=1)
    full = (1 << n) - 1
    assert abs(att.phi.sum() - (game(full) - game(0))) < 1e-10


def test_evaluation_accounting_bounded():
    n = 7
    game = random_game(n, seed=2)
    att = sampled_shapley(game, n, passes=64, seed=3)
    assert att.evaluations <= 64 * n + 1
    assert att.passes == 64


def test_reproducible_given_seed():
    n = 6
    game = random_game(n, seed=4)
    a = sampled_shapley(game, n, passes=32, seed=7)
    b = sampled_shapley(game, n, passes=32, seed=7)
    assert np.array_equal(a.phi, b.phi)


def test_pick_stability_flags_substitutes():
    """Two perfect substitutes at budgeted sampling -> unstable picks;
    a dominant player -> stable picks. The cheap_sim12 signature."""
    n = 4

    # v = 1 if player 0 OR player 1 present (perfect substitutes),
    # + 0.5 if player 2 present. k=2 picks straddle {0,2} vs {1,2}.
    values = {
        m: float(bool(m & 0b11)) + 0.5 * bool(m & 0b100) for m in range(1 << n)
    }
    sub_game = TabulatedGame(n_players=n, values=values)
    # antithetic=False: with pairing, exact substitutes tie every run and the
    # deterministic tie-break would mask the instability (see sampled.py).
    unstable = resample_pick_stability(
        sub_game, n, passes=8, k=1, n_resamples=40, seed=0, antithetic=False
    )

    # dominant player 0: picks should be near-unanimous at the same budget
    dom_values = {m: 2.0 * bool(m & 1) + 0.1 * bin(m).count("1") for m in
                  range(1 << n)}
    dom_game = TabulatedGame(n_players=n, values=dom_values)
    stable = resample_pick_stability(
        dom_game, n, passes=8, k=1, n_resamples=40, seed=0
    )

    assert stable["modal_agreement"] == 1.0
    assert unstable["modal_agreement"] < 1.0
