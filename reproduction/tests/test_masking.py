"""Integration tests against the pinned TopoBench: coalition masking, pruning,
dead-route revival, and the cell-explanation game.

These are the fresh-codebase equivalents of #13's exhaustive column-slicing
check and #1's dead-route discovery, run on the two-triangle fixture.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("topobench")
pytest.importorskip("torch_geometric")

from toposhap.neighborhoods import (  # noqa: E402
    CoalitionMaskedBackbone,
    prune_backbone_,
)
from toposhap.patches import apply_interrank_fix  # noqa: E402
from toposhap.testing import full_vocabulary_backbone, two_triangle_batch  # noqa: E402
from toposhap.vocabulary import (  # noqa: E402
    FULL_MASK,
    NEIGHBORHOODS,
    coalition_to_mask,
    mask_to_coalition,
)


@pytest.fixture(autouse=True)
def _fixed_orientation(monkeypatch):
    monkeypatch.setenv("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")
    assert apply_interrank_fix() is True
    yield


def flat(out: dict) -> torch.Tensor:
    return torch.cat([out[rank].flatten() for rank in sorted(out)])


def forward_out(backbone, batch):
    with torch.no_grad():
        return backbone(batch)


def test_full_coalition_matches_unwrapped():
    batch = two_triangle_batch()
    plain = full_vocabulary_backbone()
    ref = flat(forward_out(plain, two_triangle_batch()))

    wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone())
    out = flat(forward_out(wrapped, batch))
    assert torch.allclose(ref, out, atol=1e-6)


def test_masking_equals_subset_built_backbone():
    """Column-slicing check: masking the full backbone to coalition S must
    equal a backbone constructed with exactly S's neighborhoods (identical
    template weights). Spot-checked over structured + random masks."""
    rng = np.random.default_rng(0)
    masks = [21, 26, 72, 129, 329, 479] + [
        int(rng.integers(1, FULL_MASK + 1)) for _ in range(6)
    ]
    wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone(seed=1))
    for mask in masks:
        batch = two_triangle_batch(seed=2)
        with wrapped.coalition(mask):
            masked_out = forward_out(wrapped, batch)

        from toposhap.testing import full_vocabulary_backbone as build

        subset_names = list(mask_to_coalition(mask))
        subset_backbone = build(seed=1)
        prune_backbone_(subset_backbone, mask)
        assert subset_backbone.neighborhoods == subset_names
        subset_out = forward_out(subset_backbone, two_triangle_batch(seed=2))

        for rank in subset_out:
            assert torch.allclose(
                masked_out[rank], subset_out[rank], atol=1e-5
            ), f"mask {mask} rank {rank} diverged"


def test_prune_keeps_trained_weights():
    backbone = full_vocabulary_backbone(seed=3)
    keep = coalition_to_mask(["up_adjacency-0", "up_incidence-0"])
    ref = {
        name: p.clone()
        for name, p in backbone.graph_routes[0][0].named_parameters()
    }
    kept_names = prune_backbone_(backbone, keep)
    assert kept_names == ["up_adjacency-0", "up_incidence-0"]
    for name, p in backbone.graph_routes[0][0].named_parameters():
        assert torch.equal(p, ref[name])
    with pytest.raises(ValueError):
        prune_backbone_(backbone, 0)


def test_wrapper_rejects_reordered_vocabulary():
    backbone = full_vocabulary_backbone()
    backbone.neighborhoods = list(reversed(backbone.neighborhoods))
    with pytest.raises(ValueError, match="canonical vocabulary"):
        CoalitionMaskedBackbone(backbone)


def test_interrank_routes_dead_upstream_alive_fixed(monkeypatch):
    """The bug-1 discovery, reproduced: under upstream orientation an
    inter-rank route transmits no information from its source cells; under the
    fix it does."""
    from topobench.nn.backbones.combinatorial import gccn

    from toposhap.patches import fixed_interrank_boundary_index
    from toposhap.testing import full_vocabulary_backbone as build

    interrank_only = coalition_to_mask(["up_incidence-0"])  # nodes -> edges

    def edge_output_after_perturbing_nodes(orientation_fixed: bool):
        if orientation_fixed:
            gccn.interrank_boundary_index = fixed_interrank_boundary_index
        wrapped = CoalitionMaskedBackbone(build(seed=4))
        outs = []
        for bump in (0.0, 10.0):
            batch = two_triangle_batch(seed=5)
            batch.x_0 = batch.x_0 + bump  # perturb SOURCE cells only
            with wrapped.coalition(interrank_only), torch.no_grad():
                outs.append(wrapped(batch)[1].clone())  # dst rank = edges
        return outs

    def upstream_fn(x_src, boundary_index, n_dst_nodes):
        # verbatim upstream behaviour (pinned 6d8953e7e170)
        node_ids, edge_ids = boundary_index
        adjusted = edge_ids + n_dst_nodes
        edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
        edge_index[0, :] = node_ids
        edge_index[1, :] = adjusted
        return edge_index, x_src[edge_ids].squeeze()

    original = gccn.interrank_boundary_index
    try:
        gccn.interrank_boundary_index = upstream_fn
        dead_a, dead_b = edge_output_after_perturbing_nodes(False)
        alive_a, alive_b = edge_output_after_perturbing_nodes(True)
    finally:
        gccn.interrank_boundary_index = original

    assert torch.allclose(dead_a, dead_b, atol=1e-6), (
        "upstream orientation unexpectedly transmits source information"
    )
    assert not torch.allclose(alive_a, alive_b, atol=1e-3), (
        "fixed orientation failed to transmit source information"
    )


def test_cell_masking_game_on_backbone():
    """Leg-1 smoke: exact cell explanation on the fixture, efficiency holds."""
    from toposhap.cells import CellPlayer, explain_cells
    from toposhap.metrics import check_efficiency
    from toposhap.shapley.games import CachedGame
    from toposhap.cells.explain import CellMaskingGame

    wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone(seed=6))
    batch = two_triangle_batch(seed=7)

    def model_fn(b):
        with torch.no_grad():
            return wrapped(b)[0].sum()  # scalar readout over node rank

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
