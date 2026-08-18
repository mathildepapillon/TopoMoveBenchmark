"""Regime selection, cost caps, and diagnostics of explain_cells.

The additive fixture game (a model that sums every feature row, with a
zeros baseline) has closed-form Shapley values: phi_i is exactly row i's
feature sum, in the exact AND the sampled regime (every marginal
contribution of player i equals its row sum). That makes regime behavior
checkable without tolerance games.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from topobench.explain import (
    DEFAULT_MAX_EXACT_EVALUATIONS,
    CellPlayer,
    explain_cells,
)


def additive_parts(n_cells=17, channels=2, seed=0):
    """Build a batch, model_fn, and players for the additive game.

    Parameters
    ----------
    n_cells : int
        Number of rank-0 cells (players).
    channels : int
        Feature dimension.
    seed : int
        Seed for the random features.

    Returns
    -------
    tuple
        (batch, model_fn, players); the model sums every feature row, so
        with a zeros baseline phi_i == batch.x_0[i].sum() exactly.
    """
    g = torch.Generator().manual_seed(seed)
    batch = SimpleNamespace(x_0=torch.randn(n_cells, channels, generator=g))
    players = [CellPlayer(rank=0, index=i) for i in range(n_cells)]

    def model_fn(b):
        return b.x_0.sum()

    return batch, model_fn, players


def test_default_cap_falls_back_to_sampling_with_warning():
    """17 players exceed the default cap: sampled path plus a warning.

    2^17 = 131072 > DEFAULT_MAX_EXACT_EVALUATIONS = 2^16, so exact
    enumeration must not run silently; the warning must say how to get
    exactness back.
    """
    assert DEFAULT_MAX_EXACT_EVALUATIONS == 65536
    batch, model_fn, players = additive_parts(n_cells=17)
    with pytest.warns(UserWarning, match="max_exact_evaluations"):
        explanation = explain_cells(
            model_fn, batch, players, passes=8, seed=0, baseline="zeros"
        )
    assert not explanation.exact
    assert explanation.evaluations <= 8 * 17 + 1  # sampled-budget bound
    # additive game: sampling is exact per pass
    expected = batch.x_0.sum(dim=1).numpy()
    assert np.allclose(explanation.phi, expected, atol=1e-5)


def test_explicit_high_cap_restores_exactness():
    """Raising max_exact_evaluations to 2^n restores exact enumeration."""
    batch, model_fn, players = additive_parts(n_cells=17)
    explanation = explain_cells(
        model_fn,
        batch,
        players,
        baseline="zeros",
        max_exact_evaluations=1 << 17,
    )
    assert explanation.exact
    assert explanation.evaluations == 1 << 17
    expected = batch.x_0.sum(dim=1).numpy()
    assert np.allclose(explanation.phi, expected, atol=1e-5)


def test_within_cap_stays_exact_and_silent():
    """Under the cap the exact regime runs without any warning."""
    import warnings

    batch, model_fn, players = additive_parts(n_cells=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explanation = explain_cells(model_fn, batch, players, baseline="zeros")
    assert explanation.exact
    assert explanation.evaluations == 1 << 6


def test_interactions_refused_when_cap_forces_sampling():
    """Interactions need the exact regime; the cap must be named."""
    batch, model_fn, players = additive_parts(n_cells=17)
    with pytest.raises(ValueError, match="max_exact_evaluations"):
        explain_cells(
            model_fn, batch, players, interactions=True, baseline="zeros"
        )
