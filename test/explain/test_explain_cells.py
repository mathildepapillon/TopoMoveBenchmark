"""Regime selection, cost caps, and diagnostics of explain_cells.

The additive fixture game (a model that sums every feature row, with a
zeros baseline) has closed-form Shapley values: phi_i is exactly row i's
feature sum, in the exact AND the sampled regime (every marginal
contribution of player i equals its row sum). That makes regime behavior
checkable without tolerance games.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from topobench.explain import (
    DEFAULT_MAX_EXACT_EVALUATIONS,
    CellMaskingGame,
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


def test_constant_game_diagnostic():
    """A structurally inert game must warn; a live game must not.

    A HOPSE-style model reads only the per-hop tensors ``x{rank}_{hop}``,
    so the plain game's masking of ``x_{rank}`` never reaches the
    computation: v(full) == v(empty) exactly and every Shapley value is
    0. explain_cells must warn loudly, naming HopseCellMaskingGame as the
    fix; a model that does read ``x_{rank}`` must run warning-free.
    """
    from test.explain.test_hopse_cell_masking import (
        hopse_batch,
        make_game_parts,
    )

    encoder, model_fn, players = make_game_parts()
    with pytest.warns(UserWarning, match="HopseCellMaskingGame"):
        explanation = explain_cells(
            model_fn, hopse_batch(seed=3), players, encoder=encoder
        )
    assert np.allclose(explanation.phi, 0.0)

    # a normal model (reads x_0) does not trigger the diagnostic
    batch, sum_model_fn, sum_players = additive_parts(n_cells=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explain_cells(sum_model_fn, batch, sum_players, baseline="zeros")


def test_baseline_dict_with_non_player_rank_names_valid_ranks():
    """A baseline keyed by a non-player rank must fail instructively."""
    batch, model_fn, players = additive_parts(n_cells=4)
    with pytest.raises(ValueError, match=r"valid ranks are \[0\]"):
        CellMaskingGame(
            model_fn, batch, players, baseline={1: torch.zeros(2)}
        )


def test_wrong_width_baseline_is_refused_at_construction():
    """Shape mismatches must fail at construction, naming the widths."""
    batch, model_fn, players = additive_parts(n_cells=4, channels=3)
    # one row per rank, wrong width
    with pytest.raises(ValueError, match="width 5.*width 3"):
        CellMaskingGame(
            model_fn, batch, players, baseline={0: torch.ones(5)}
        )
    # per-cell matrix, wrong shape
    with pytest.raises(ValueError, match=r"shape \(2, 3\).*shape \(4, 3\)"):
        CellMaskingGame(
            model_fn, batch, players, baseline={0: torch.ones(2, 3)}
        )
    # unsupported dimensionality
    with pytest.raises(ValueError, match="3-d tensor"):
        CellMaskingGame(
            model_fn, batch, players, baseline={0: torch.ones(4, 3, 1)}
        )


def test_batch_without_any_rank_features_says_exactly_that():
    """No x_{rank} attributes at all must produce a precise message."""
    batch = SimpleNamespace(edge_index=torch.zeros(2, 0, dtype=torch.long))
    players = [CellPlayer(rank=0, index=0)]
    with pytest.raises(
        ValueError, match="no x_.rank. feature matrices at all"
    ):
        CellMaskingGame(lambda b: 0.0, batch, players)


def test_rank_discovery_works_without_rank_zero_features():
    """Players on an existing rank must work even when x_0 is absent.

    Rank discovery used to walk x_0, x_1, ... from zero, so a batch with
    only x_1 produced the false message "batch has no features for ranks
    {1}" although x_1 existed. Ranks are now also discovered from the
    players themselves.
    """
    g = torch.Generator().manual_seed(0)
    batch = SimpleNamespace(x_1=torch.randn(5, 2, generator=g))
    players = [CellPlayer(rank=1, index=i) for i in range(5)]

    def model_fn(b):
        return b.x_1.sum()

    explanation = explain_cells(model_fn, batch, players, baseline="zeros")
    assert explanation.exact
    expected = batch.x_1.sum(dim=1).numpy()
    assert np.allclose(explanation.phi, expected, atol=1e-5)


def test_missing_player_rank_error_names_available_ranks():
    """A player on a genuinely absent rank must name what IS available."""
    batch, model_fn, _ = additive_parts(n_cells=4)
    players = [CellPlayer(rank=0, index=0), CellPlayer(rank=1, index=0)]
    with pytest.raises(
        ValueError, match=r"ranks \[1\].*available ranks: \[0\]"
    ):
        CellMaskingGame(model_fn, batch, players)
