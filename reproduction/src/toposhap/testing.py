"""Synthetic combinatorial-complex fixtures for tests and smoke gates.

The canonical fixture is two triangles sharing an edge, lifted to a 2-complex:
4 nodes, 5 edges, 2 faces. Every neighborhood in the 11-vocabulary is
non-trivial on it, so it exercises all routes of a full-vocabulary TopoTune
backbone (the same role #13's exhaustive column-slicing check played for
HOPSE). Requires torch; import only inside torch-dependent code.

Neighborhood matrices follow the TopoBench convention established by
``get_nbhd_cache``/``interrank_boundary_index``: sparse shape
``[n_dst, n_src]`` with indices ``(dst_cell, src_cell)``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from toposhap.vocabulary import NEIGHBORHOODS

# two triangles sharing edge (1,2): A = {0,1,2}, B = {1,2,3}
_NODES = [0, 1, 2, 3]
_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]  # edge ids 0..4
_FACES = [(0, 1, 2), (1, 2, 3)]  # face ids 0..1
_FACE_EDGES = {0: [0, 1, 2], 1: [2, 3, 4]}


def _sparse(pairs: list[tuple[int, int]], shape: tuple[int, int]):
    if not pairs:
        idx = torch.zeros((2, 0), dtype=torch.long)
        vals = torch.zeros((0,))
    else:
        idx = torch.tensor(pairs, dtype=torch.long).T
        vals = torch.ones(len(pairs))
    return torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()


def two_triangle_batch(channels: int = 8, seed: int = 0) -> SimpleNamespace:
    """A single-complex batch object duck-typing what TopoTune.forward needs."""
    g = torch.Generator().manual_seed(seed)
    n0, n1, n2 = len(_NODES), len(_EDGES), len(_FACES)

    node_pairs_via_edge = []
    for a, b in _EDGES:
        node_pairs_via_edge += [(a, b), (b, a)]

    edge_pairs_coface = []
    for edges in _FACE_EDGES.values():
        for e in edges:
            for f in edges:
                if e != f:
                    edge_pairs_coface.append((e, f))
    edge_pairs_coface = sorted(set(edge_pairs_coface))

    edge_pairs_shared_node = sorted(
        {
            (e, f)
            for e, (a, b) in enumerate(_EDGES)
            for f, (c, d) in enumerate(_EDGES)
            if e != f and {a, b} & {c, d}
        }
    )

    node_pairs_via_face = sorted(
        {
            (a, b)
            for face in _FACES
            for a in face
            for b in face
            if a != b
        }
    )

    edge_node = [(e, n) for e, (a, b) in enumerate(_EDGES) for n in (a, b)]
    node_edge = [(n, e) for e, n in edge_node]
    edge_face = [(e, f) for f, edges in _FACE_EDGES.items() for e in edges]
    face_edge = [(f, e) for e, f in edge_face]
    face_node = [(f, n) for f, face in enumerate(_FACES) for n in face]
    node_face = [(n, f) for f, n in face_node]
    face_pairs_shared_edge = [(0, 1), (1, 0)]

    mats = {
        "up_adjacency-0": _sparse(node_pairs_via_edge, (n0, n0)),
        "up_incidence-0": _sparse(edge_node, (n1, n0)),
        "down_incidence-1": _sparse(node_edge, (n0, n1)),
        "up_adjacency-1": _sparse(edge_pairs_coface, (n1, n1)),
        "down_incidence-2": _sparse(edge_face, (n1, n2)),
        "2-up_adjacency-0": _sparse(node_pairs_via_face, (n0, n0)),
        "down_adjacency-1": _sparse(edge_pairs_shared_node, (n1, n1)),
        "up_incidence-1": _sparse(face_edge, (n2, n1)),
        "down_adjacency-2": _sparse(face_pairs_shared_edge, (n2, n2)),
        "2-up_incidence-0": _sparse(face_node, (n2, n0)),
        "2-down_incidence-2": _sparse(node_face, (n0, n2)),
    }
    assert set(mats) == set(NEIGHBORHOODS)

    batch = SimpleNamespace(
        x_0=torch.randn(n0, channels, generator=g),
        incidence_1=_sparse(node_edge, (n0, n1)),
        incidence_2=_sparse(edge_face, (n1, n2)),
        x_1=torch.randn(n1, channels, generator=g),
        x_2=torch.randn(n2, channels, generator=g),
        cell_statistics=torch.tensor([[n0, n1, n2]], dtype=torch.long),
        **mats,
    )
    return batch


def full_vocabulary_backbone(channels: int = 8, layers: int = 1, seed: int = 0):
    """A TopoTune backbone over the full 11-neighborhood vocabulary.

    Route GNNs are deepcopies of one template, so a backbone built with a
    subset vocabulary from the same seed has identical weights on shared
    routes — the property the coalition-masking equivalence tests exploit.
    """
    from torch_geometric.nn.models import GCN

    from topobench.nn.backbones.combinatorial.gccn import TopoTune

    torch.manual_seed(seed)
    gnn = GCN(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        num_layers=1,
    )
    backbone = TopoTune(
        GNN=gnn,
        neighborhoods=list(NEIGHBORHOODS),
        layers=layers,
        use_edge_attr=False,
        activation="relu",
    )
    backbone.eval()
    return backbone


def two_triangle_participation_batch(channels: int = 8, seed: int = 0):
    """Fixture whose neighborhood matrices follow the REAL lift recipe.

    ``CellParticipationGame`` rebuilds neighborhood matrices from the
    incidences with the production conventions (signed boundary
    products, diagonals and cancelled zeros kept, the selector's 2-hop
    compositions). The hand-built matrices of ``two_triangle_batch``
    use a simplified unit convention, so participation tests use this
    recipe-derived variant — self-consistent with the rebuild by
    construction.
    """
    from topobench.data.utils.utils import select_neighborhoods_of_interest

    from toposhap.cells.participation import _base_connectivity

    base = two_triangle_batch(channels=channels, seed=seed)
    mats = select_neighborhoods_of_interest(
        _base_connectivity(
            base.incidence_1.coalesce(), base.incidence_2.coalesce()
        ),
        list(NEIGHBORHOODS),
    )
    for name, mat in mats.items():
        setattr(base, name, mat.coalesce())
    return base
