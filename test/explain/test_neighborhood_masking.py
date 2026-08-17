"""Integration tests for coalition masking on TopoTune backbones.

Masking a backbone to a coalition must be indistinguishable from building
the backbone with exactly that coalition's neighborhoods (an exhaustive
route-slicing check on structured and random masks), pruning must keep
trained weights, and the cell-explanation game must satisfy efficiency on
a real backbone.
"""

import numpy as np
import pytest
import torch

from test.explain.fixtures import (
    NEIGHBORHOODS,
    build_backbone,
    randomize_route_weights_,
    two_triangle_batch,
)
from topobench.explain import (
    CellPlayer,
    CoalitionMaskedBackbone,
    check_efficiency,
    coalition_to_mask,
    explain_cells,
    mask_to_coalition,
    prune_backbone_,
)
from topobench.explain.cells import CellMaskingGame
from topobench.explain.games import CachedGame

N = len(NEIGHBORHOODS)
FULL_MASK = (1 << N) - 1


def make_backbone(layers=1, template_seed=1, weight_seed=2):
    """Build the fixture backbone with distinct per-route weights.

    Parameters
    ----------
    layers : int
        Number of TopoTune layers.
    template_seed : int
        Seed for the GNN template.
    weight_seed : int
        Seed for the per-route weight randomization.

    Returns
    -------
    TopoTune
        Deterministically constructed backbone.
    """
    backbone = build_backbone(
        NEIGHBORHOODS, layers=layers, seed=template_seed
    )
    randomize_route_weights_(backbone, seed=weight_seed)
    return backbone


def subset_built_backbone(full, mask, layers=1):
    """Build a backbone with only the coalition's neighborhoods.

    The subset backbone is constructed independently (not pruned from
    ``full``) and its route GNNs load the weights of the corresponding
    routes of ``full``, so it is the ground truth for what masking should
    compute.

    Parameters
    ----------
    full : TopoTune
        Backbone over the full neighborhood list.
    mask : int
        Coalition bitmask over the full neighborhood list.
    layers : int
        Number of TopoTune layers.

    Returns
    -------
    TopoTune
        Subset backbone with weights copied from the full backbone.
    """
    kept = [i for i in range(N) if mask >> i & 1]
    sub = build_backbone(
        [NEIGHBORHOODS[i] for i in kept], layers=layers, seed=1
    )
    for layer_idx in range(layers):
        for new_idx, old_idx in enumerate(kept):
            sub.graph_routes[layer_idx][new_idx].load_state_dict(
                full.graph_routes[layer_idx][old_idx].state_dict()
            )
    return sub


def forward_out(backbone, batch):
    """Run a backbone without gradients.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone (wrapped or not).
    batch : types.SimpleNamespace
        Fixture batch.

    Returns
    -------
    dict
        Per-rank outputs.
    """
    with torch.no_grad():
        return backbone(batch)


def test_wrapper_captures_players_in_bit_order():
    """The wrapper must expose the configured neighborhood list as players."""
    wrapped = CoalitionMaskedBackbone(make_backbone())
    assert wrapped.players == NEIGHBORHOODS
    assert wrapped.full_mask == FULL_MASK
    assert wrapped.coalition_mask == FULL_MASK

    # any list length works: player order is just the configured order
    short = ["up_incidence-0", "down_incidence-1"]
    wrapped_short = CoalitionMaskedBackbone(build_backbone(short, seed=3))
    assert wrapped_short.players == short
    assert wrapped_short.full_mask == 0b11


def test_mask_name_round_trip():
    """coalition_to_mask and mask_to_coalition must be mutually inverse."""
    names = ("up_incidence-0", "down_incidence-1")
    mask = coalition_to_mask(names, NEIGHBORHOODS)
    assert mask == (1 << 1) | (1 << 4)
    assert mask_to_coalition(mask, NEIGHBORHOODS) == names

    with pytest.raises(ValueError, match="out of range"):
        mask_to_coalition(1 << N, NEIGHBORHOODS)
    with pytest.raises(ValueError, match="unknown neighborhood"):
        coalition_to_mask(["nope"], NEIGHBORHOODS)


def test_full_coalition_matches_unwrapped():
    """Under the grand coalition the wrapper must equal the plain backbone."""
    plain = make_backbone()
    ref = forward_out(plain, two_triangle_batch(seed=0))

    wrapped = CoalitionMaskedBackbone(make_backbone())
    out = forward_out(wrapped, two_triangle_batch(seed=0))
    for rank in ref:
        assert torch.allclose(ref[rank], out[rank], atol=1e-6)


@pytest.mark.parametrize("layers", [1, 2])
def test_masking_equals_subset_built_backbone(layers):
    """Route-slicing check: masking == a backbone built with the subset.

    Masking the full backbone to coalition S must equal a backbone
    constructed with exactly S's neighborhoods and the same per-route
    weights. Checked on structured and random masks.
    """
    rng = np.random.default_rng(0)
    masks = [
        FULL_MASK,
        0b000001,  # single intra-rank route
        coalition_to_mask(
            ["up_incidence-0", "down_incidence-1"], NEIGHBORHOODS
        ),  # inter-rank routes only
    ] + [int(rng.integers(1, FULL_MASK + 1)) for _ in range(5)]

    full = make_backbone(layers=layers)
    wrapped = CoalitionMaskedBackbone(make_backbone(layers=layers))
    for mask in masks:
        with wrapped.coalition(mask):
            masked_out = forward_out(wrapped, two_triangle_batch(seed=2))

        sub = subset_built_backbone(full, mask, layers=layers)
        assert sub.neighborhoods == list(
            mask_to_coalition(mask, NEIGHBORHOODS)
        )
        subset_out = forward_out(sub, two_triangle_batch(seed=2))

        for rank in subset_out:
            assert torch.allclose(
                masked_out[rank], subset_out[rank], atol=1e-5
            ), f"mask {mask} rank {rank} diverged"


def test_coalition_context_restores_mask():
    """The coalition context manager must restore the previous mask."""
    wrapped = CoalitionMaskedBackbone(make_backbone())
    with wrapped.coalition(0b1):
        assert wrapped.coalition_mask == 0b1
    assert wrapped.coalition_mask == FULL_MASK
    with pytest.raises(ValueError, match="out of range"):
        with wrapped.coalition(FULL_MASK + 1):
            pass


def test_prune_keeps_trained_weights():
    """Pruning must keep the kept routes' weights bit for bit."""
    backbone = make_backbone(template_seed=3, weight_seed=4)
    keep_names = ["up_incidence-0", "down_incidence-1"]
    keep = coalition_to_mask(keep_names, NEIGHBORHOODS)
    kept_indices = [i for i in range(N) if keep >> i & 1]
    ref = {
        old_idx: {
            name: p.clone()
            for name, p in backbone.graph_routes[0][
                old_idx
            ].named_parameters()
        }
        for old_idx in kept_indices
    }

    kept_names = prune_backbone_(backbone, keep)
    assert kept_names == keep_names
    assert backbone.neighborhoods == keep_names
    assert len(backbone.graph_routes[0]) == len(keep_names)
    for new_idx, old_idx in enumerate(kept_indices):
        for name, p in backbone.graph_routes[0][
            new_idx
        ].named_parameters():
            assert torch.equal(p, ref[old_idx][name])

    with pytest.raises(ValueError, match="empty coalition"):
        prune_backbone_(backbone, 0)


def test_pruned_backbone_matches_masked_forward():
    """A pruned backbone must compute what the mask computed."""
    mask = coalition_to_mask(
        ["up_adjacency-1", "up_incidence-0", "2-up_adjacency-0"],
        NEIGHBORHOODS,
    )
    wrapped = CoalitionMaskedBackbone(make_backbone())
    with wrapped.coalition(mask):
        masked_out = forward_out(wrapped, two_triangle_batch(seed=5))

    pruned = make_backbone()
    prune_backbone_(pruned, mask)
    pruned_out = forward_out(pruned, two_triangle_batch(seed=5))
    for rank in pruned_out:
        assert torch.allclose(
            masked_out[rank], pruned_out[rank], atol=1e-5
        )


def test_cell_masking_game_efficiency_on_backbone():
    """Exact cell explanation on a real backbone must satisfy efficiency."""
    backbone = make_backbone(template_seed=6, weight_seed=7)
    batch = two_triangle_batch(seed=7)

    def model_fn(b):
        with torch.no_grad():
            return backbone(b)[0].sum()  # scalar readout over node rank

    players = [CellPlayer(rank=0, index=i) for i in range(4)] + [
        CellPlayer(rank=2, index=j) for j in range(2)
    ]
    explanation = explain_cells(model_fn, batch, players, interactions=True)
    assert explanation.exact
    assert explanation.evaluations == 1 << len(players)

    game = CachedGame(
        n_players=len(players),
        evaluate=CellMaskingGame(model_fn, batch, players),
    )
    check_efficiency(explanation.phi, game, len(players), atol=1e-4)


class _RankEncoder(torch.nn.Module):
    """Per-rank linear feature encoder (mutates the batch and returns it)."""

    def __init__(self, channels, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.linears = torch.nn.ModuleDict(
            {str(r): torch.nn.Linear(channels, channels) for r in range(3)}
        )

    def forward(self, batch):
        for r in range(3):
            setattr(
                batch,
                f"x_{r}",
                self.linears[str(r)](getattr(batch, f"x_{r}")),
            )
        return batch


def test_cell_game_encoded_masking_baselines():
    """With an encoder the game masks encoded rows; masked rows become the
    chosen baseline (per-rank mean by default, one row per rank, per-cell
    rows, or zeros), and the batch is restored after every evaluation."""
    backbone = make_backbone(template_seed=8, weight_seed=9)
    encoder = _RankEncoder(channels=8, seed=10)

    def model_fn(b):
        with torch.no_grad():
            return backbone(b)[0].sum()  # scalar readout over node rank

    players = [CellPlayer(rank=0, index=i) for i in range(4)] + [
        CellPlayer(rank=2, index=j) for j in range(2)
    ]

    def fresh_encoded():
        with torch.no_grad():
            return encoder(two_triangle_batch(seed=11))

    def value_with_rows(fill_by_rank):
        b = fresh_encoded()
        for rank, fill in fill_by_rank.items():
            x = getattr(b, f"x_{rank}")
            setattr(b, f"x_{rank}", fill.expand_as(x).clone())
        with torch.no_grad():
            return float(model_fn(b))

    # default with an encoder: per-rank mean of the encoded rows
    game = CellMaskingGame(
        model_fn, two_triangle_batch(seed=11), players, encoder=encoder
    )
    assert game.baseline_name == "complex_mean"
    reference = fresh_encoded()
    expected = value_with_rows(
        {0: reference.x_0.mean(0), 2: reference.x_2.mean(0)}
    )
    assert abs(game(0) - expected) < 1e-5

    # snapshot/restore: the encoded originals survive evaluations
    for rank in range(3):
        assert torch.equal(
            getattr(game.batch, f"x_{rank}"), game._originals[rank]
        )

    # one baseline row per rank (e.g. train-split means)
    rows = {
        0: torch.randn(8, generator=torch.Generator().manual_seed(12)),
        2: torch.randn(8, generator=torch.Generator().manual_seed(13)),
    }
    game_rows = CellMaskingGame(
        model_fn,
        two_triangle_batch(seed=11),
        players,
        baseline=dict(rows),
        encoder=encoder,
    )
    assert abs(game_rows(0) - value_with_rows(rows)) < 1e-5

    # per-cell replacement rows keep working
    mats = {
        0: torch.randn(4, 8, generator=torch.Generator().manual_seed(14)),
        2: torch.randn(2, 8, generator=torch.Generator().manual_seed(15)),
    }
    game_mats = CellMaskingGame(
        model_fn,
        two_triangle_batch(seed=11),
        players,
        baseline=dict(mats),
        encoder=encoder,
    )
    b = fresh_encoded()
    b.x_0 = mats[0].clone()
    b.x_2 = mats[2].clone()
    with torch.no_grad():
        expected_mats = float(model_fn(b))
    assert abs(game_mats(0) - expected_mats) < 1e-5

    # zeros stays available with an encoder
    game_zero = CellMaskingGame(
        model_fn,
        two_triangle_batch(seed=11),
        players,
        baseline="zeros",
        encoder=encoder,
    )
    expected_zero = value_with_rows({0: torch.zeros(8), 2: torch.zeros(8)})
    assert abs(game_zero(0) - expected_zero) < 1e-5

    # the full coalition is encoder-only: identical across all baselines
    full = (1 << len(players)) - 1
    for g in (game_rows, game_mats, game_zero):
        assert abs(g(full) - game(full)) < 1e-5


def test_cell_game_encoded_efficiency():
    """explain_cells with an encoder satisfies efficiency on the same game."""
    backbone = make_backbone(template_seed=8, weight_seed=9)
    encoder = _RankEncoder(channels=8, seed=10)

    def model_fn(b):
        with torch.no_grad():
            return backbone(b)[0].sum()

    players = [CellPlayer(rank=0, index=i) for i in range(4)] + [
        CellPlayer(rank=2, index=j) for j in range(2)
    ]
    explanation = explain_cells(
        model_fn, two_triangle_batch(seed=11), players, encoder=encoder
    )
    assert explanation.exact
    game = CachedGame(
        n_players=len(players),
        evaluate=CellMaskingGame(
            model_fn, two_triangle_batch(seed=11), players, encoder=encoder
        ),
    )
    check_efficiency(explanation.phi, game, len(players), atol=1e-4)
