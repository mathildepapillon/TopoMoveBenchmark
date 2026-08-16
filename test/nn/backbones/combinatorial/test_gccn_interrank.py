"""Regression tests for the inter-rank edge_index orientation in GCCN.

``interrank_boundary_index`` must orient the lifted edge_index
source-to-target (PyG convention): row 0 holds the offset source-cell ids
and row 1 the destination-cell ids. ``interrank_gnn_forward`` reads the
first ``n_dst_cells`` rows of the GNN output, so with the reversed
orientation inter-rank messages are aggregated onto the offset source
slots — which are never read — and inter-rank routes transmit no
information from their source cells.
"""

import torch
from torch_geometric.nn.models import GCN

from topobench.nn.backbones.combinatorial.gccn import (
    TopoTune,
    interrank_boundary_index,
)

# Two triangles sharing an edge: 4 nodes, 5 edges (a 1-complex is enough
# for a single node->edge incidence route).
_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]


def _incidence_batch(channels=8, seed=0, bump=0.0):
    """Build a minimal batch with a node->edge incidence neighborhood.

    Parameters
    ----------
    channels : int
        Feature dimension of every rank.
    seed : int
        Seed for the random features.
    bump : float
        Constant added to the source-rank (node) features only.

    Returns
    -------
    torch_geometric.data.Data
        Batch exposing x_0, x_1, cell_statistics and 'up_incidence-0'.
    """
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    n0, n1 = 4, len(_EDGES)
    # sparse [n_dst, n_src] with indices (dst_cell, src_cell)
    pairs = [(e, n) for e, (a, b) in enumerate(_EDGES) for n in (a, b)]
    idx = torch.tensor(pairs, dtype=torch.long).T
    incidence = torch.sparse_coo_tensor(
        idx, torch.ones(len(pairs)), size=(n1, n0)
    ).coalesce()

    batch = Data(
        x_0=torch.randn(n0, channels, generator=g) + bump,
        x_1=torch.randn(n1, channels, generator=g),
        cell_statistics=torch.tensor([[n0, n1]], dtype=torch.long),
    )
    batch["up_incidence-0"] = incidence
    return batch


def _interrank_only_backbone(channels=8, seed=0):
    """Build a TopoTune backbone with a single inter-rank route.

    Parameters
    ----------
    channels : int
        Feature dimension of every rank.
    seed : int
        Seed for the GNN template initialization.

    Returns
    -------
    TopoTune
        Backbone whose only route is nodes -> edges ('up_incidence-0').
    """
    torch.manual_seed(seed)
    gnn = GCN(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        num_layers=1,
    )
    backbone = TopoTune(
        GNN=gnn,
        neighborhoods=["up_incidence-0"],
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )
    backbone.eval()
    return backbone


def test_interrank_edge_index_targets_destination_cells():
    """Row 1 (targets) must hold destination ids, row 0 offset source ids."""
    n_dst, n_src = 4, 5
    x_src = torch.randn(n_src, 8)
    # boundary_index[0] = destination-cell ids, [1] = source-cell ids
    dst_ids = torch.tensor([0, 1, 2, 3, 0, 2])
    src_ids = torch.tensor([0, 0, 1, 3, 4, 2])

    edge_index, edge_attr = interrank_boundary_index(
        x_src, [dst_ids, src_ids], n_dst
    )

    # targets are destination cells: within [0, n_dst)
    assert torch.equal(edge_index[1], dst_ids)
    assert bool((edge_index[1] < n_dst).all())
    # sources are the offset source cells: within [n_dst, n_dst + n_src)
    assert torch.equal(edge_index[0], src_ids + n_dst)
    assert bool((edge_index[0] >= n_dst).all())
    # edge features come from the source cells
    assert torch.allclose(edge_attr, x_src[src_ids])


def test_interrank_route_transmits_source_information():
    """Perturbing source-rank features must change destination-rank output.

    An inter-rank-only backbone (nodes -> edges) is evaluated twice, once
    with the node features shifted by a large constant. If the edge-rank
    output does not move, the route is dead — the failure mode of the
    reversed edge_index orientation.
    """
    backbone = _interrank_only_backbone(seed=4)
    outs = []
    for bump in (0.0, 10.0):
        batch = _incidence_batch(seed=5, bump=bump)
        with torch.no_grad():
            outs.append(backbone(batch)[1].clone())  # dst rank = edges
    assert not torch.allclose(outs[0], outs[1], atol=1e-3), (
        "inter-rank route transmitted no information from its source cells"
    )


def test_interrank_output_lands_on_destination_rows():
    """Messages must reach the rows read by interrank_gnn_forward.

    With linear (identity-activation) aggregation and zero-initialized
    destination features, the destination-rank output must differ from a
    forward pass in which the incidence structure is empty; an empty
    neighborhood leaves destinations with only self-loop contributions.
    """
    backbone = _interrank_only_backbone(seed=1)

    batch = _incidence_batch(seed=2)
    with torch.no_grad():
        out_full = backbone(batch)[1].clone()

    # same batch but with an empty node->edge incidence
    empty = _incidence_batch(seed=2)
    n0, n1 = 4, len(_EDGES)
    empty["up_incidence-0"] = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), size=(n1, n0)
    ).coalesce()
    with torch.no_grad():
        out_empty = backbone(empty)[1].clone()

    assert not torch.allclose(out_full, out_empty, atol=1e-3), (
        "incidence structure had no effect on destination-rank output"
    )
