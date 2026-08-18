"""Semantic tests of the executable axiom checks.

Each check is exercised on games where the ground truth is known by
construction: a weighted-sum game has exactly the null players whose
weight is zero and exactly the symmetric pairs whose weights are equal,
and a model that subtracts the batch mean is batch-dependent by
construction while a row-wise model is not. Detection AND non-detection
are asserted — a check that flags everything (or nothing) is worthless.
"""

from types import SimpleNamespace

import pytest
import torch

from topobench.explain import (
    check_batch_invariance,
    check_efficiency,
    check_symmetry_pair,
    find_null_players,
    shapley_values,
)


def weighted_sum_game(weights):
    """Additive game: v(mask) = sum of weights of the present players.

    Player i is a null player iff ``weights[i] == 0`` (its marginal
    contribution is ``weights[i]`` on EVERY coalition), and players i, j
    are symmetric iff ``weights[i] == weights[j]``.

    Parameters
    ----------
    weights : list of float
        Per-player weight.

    Returns
    -------
    callable
        Game over coalition bitmasks.
    """

    def v(mask):
        return sum(w for i, w in enumerate(weights) if mask >> i & 1)

    return v


# ------------------------------------------------------------ null players


def test_find_null_players_detects_exactly_the_zero_weight_players():
    """Exhaustive probing finds the null players and only them."""
    game = weighted_sum_game([1.0, 0.0, 2.0, 0.0])
    assert find_null_players(game, 4) == [1, 3]


def test_find_null_players_on_a_nonlinear_game():
    """A player the value never depends on is null even under interactions.

    v(S) = 1 if both 0 and 2 are in S else 0: players 0 and 2 interact,
    player 1 is null. The interaction must not confuse detection in
    either direction.
    """

    def v(mask):
        return float((mask & 0b101) == 0b101)

    assert find_null_players(v, 3) == [1]


def test_find_null_players_sampled_probes_agree_on_additive_games():
    """The n_probes regime finds the same nulls when marginals are global.

    On an additive game a player's marginal contribution is its weight on
    every coalition, so any probe set gives the exhaustive answer — the
    sampled regime must return it too.
    """
    game = weighted_sum_game([0.5, 0.0, -1.0, 0.0, 3.0])
    assert find_null_players(game, 5, n_probes=8) == [1, 3]


def test_null_player_gets_zero_shapley_value():
    """The detected null player's exact Shapley value is exactly 0."""
    weights = [1.0, 0.0, 2.0]
    game = weighted_sum_game(weights)
    phi = shapley_values(game, 3)
    assert phi[1] == 0.0
    assert phi[0] != 0.0 and phi[2] != 0.0
    check_efficiency(phi, game, 3, atol=1e-10)


# ---------------------------------------------------------------- symmetry


def test_symmetry_pair_zero_iff_weights_equal():
    """Equal-weight players are symmetric; the asymmetry of unequal ones
    equals the weight gap exactly on an additive game."""
    game = weighted_sum_game([1.0, 1.0, 2.5])
    assert check_symmetry_pair(game, 3, 0, 1) == 0.0
    assert check_symmetry_pair(game, 3, 0, 2) == pytest.approx(1.5)
    assert check_symmetry_pair(game, 3, 1, 2) == pytest.approx(1.5)


def test_symmetry_pair_on_a_purely_cardinal_game():
    """A game depending only on |S| makes every pair symmetric."""

    def v(mask):
        return bin(mask).count("1") ** 2

    for i in range(3):
        for j in range(i + 1, 3):
            assert check_symmetry_pair(v, 3, i, j) == 0.0


def test_symmetric_players_get_equal_shapley_values():
    """The symmetry axiom holds for the exact values on a symmetric pair."""
    game = weighted_sum_game([1.0, 1.0, 2.5])
    phi = shapley_values(game, 3)
    assert phi[0] == pytest.approx(phi[1])
    assert phi[2] != pytest.approx(phi[0])


# -------------------------------------------------------- batch invariance


class _RowwiseModel(torch.nn.Module):
    """Batch-invariant by construction: each row is mapped independently."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, batch):
        return torch.tanh(self.linear(batch.x))[: batch.n_shared]


class _BatchMeanModel(torch.nn.Module):
    """Batch-dependent by construction: subtracts the batch mean."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return (batch.x - batch.x.mean(dim=0))[: batch.n_shared]


class _DictModel(torch.nn.Module):
    """Mapping output: 'x_0' is row-wise, 'aux' depends on composition."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return {
            "x_0": 2.0 * batch.x[: batch.n_shared],
            "aux": batch.x.mean(dim=0, keepdim=True),
        }


def compositions(n_shared=3, channels=3, n_extra=(0, 2, 5), seed=0):
    """Batches sharing the same first ``n_shared`` rows, padded differently.

    Parameters
    ----------
    n_shared : int
        Number of rows identical across the compositions (the samples
        whose outputs are compared).
    channels : int
        Feature dimension.
    n_extra : tuple of int
        Number of distractor rows appended in each composition.
    seed : int
        Seed for the shared rows and the distractors.

    Returns
    -------
    list of types.SimpleNamespace
        One batch per composition, each with ``x`` and ``n_shared``.
    """
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n_shared, channels, generator=g)
    batches = []
    for k in n_extra:
        extra = torch.randn(k, channels, generator=g)
        batches.append(
            SimpleNamespace(
                x=torch.cat([shared, extra]), n_shared=n_shared
            )
        )
    return batches


def test_batch_invariance_passes_on_a_rowwise_model():
    """A row-wise model's per-sample outputs ignore batch composition."""
    check_batch_invariance(_RowwiseModel(), compositions(), atol=1e-6)


def test_batch_invariance_detects_a_batch_dependent_model():
    """A model that subtracts the batch mean must be flagged, with the
    deviation in the message."""
    with pytest.raises(AssertionError, match="batch-invariance violated"):
        check_batch_invariance(_BatchMeanModel(), compositions(), atol=1e-6)


def test_batch_invariance_key_selects_the_compared_entry():
    """With key='x_0' only the row-wise entry is compared (passes); without
    a key every entry is compared and the composition-dependent 'aux'
    triggers the assertion."""
    model = _DictModel()
    check_batch_invariance(model, compositions(), atol=1e-6, key="x_0")
    with pytest.raises(AssertionError, match="batch-invariance violated"):
        check_batch_invariance(model, compositions(), atol=1e-6)


def test_batch_invariance_restores_training_mode():
    """The check evaluates in eval mode but restores train mode after."""
    model = _RowwiseModel()
    model.train()
    check_batch_invariance(model, compositions(), atol=1e-6)
    assert model.training
    model.eval()
    check_batch_invariance(model, compositions(), atol=1e-6)
    assert not model.training
