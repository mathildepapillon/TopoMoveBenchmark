"""Synthetic combinatorial-complex fixtures for the explain tests.

The canonical fixture is two triangles sharing an edge, lifted to a
2-complex: 4 nodes, 5 edges, 2 faces. Every supported neighborhood is
non-trivial on it, so it exercises intra-rank and inter-rank routes of a
TopoTune backbone with any neighborhood list drawn from the matrices below.

Neighborhood matrices follow the TopoBench convention established by
``get_nbhd_cache``/``interrank_boundary_index``: sparse shape
``[n_dst, n_src]`` with indices ``(dst_cell, src_cell)``.
"""

from types import SimpleNamespace

import torch

# two triangles sharing edge (1,2): A = {0,1,2}, B = {1,2,3}
_NODES = [0, 1, 2, 3]
_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]  # edge ids 0..4
_FACES = [(0, 1, 2), (1, 2, 3)]  # face ids 0..1
_FACE_EDGES = {0: [0, 1, 2], 1: [2, 3, 4]}

#: A generic neighborhood list: the four neighborhoods of the default
#: TopoTune config (configs/model/combinatorial/topotune.yaml) plus two
#: more inter-rank routes (edges -> nodes and edges -> faces). Player/bit
#: order is the order of this list.
NEIGHBORHOODS = [
    "up_adjacency-1",
    "up_incidence-0",
    "down_incidence-2",
    "2-up_adjacency-0",
    "down_incidence-1",
    "up_incidence-1",
]


def _sparse(pairs, shape):
    """Build a coalesced sparse COO matrix from (dst, src) index pairs.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Nonzero positions as (dst_cell, src_cell) pairs.
    shape : tuple[int, int]
        Matrix shape [n_dst, n_src].

    Returns
    -------
    torch.Tensor
        Coalesced sparse COO tensor with unit values.
    """
    if not pairs:
        idx = torch.zeros((2, 0), dtype=torch.long)
        vals = torch.zeros((0,))
    else:
        idx = torch.tensor(pairs, dtype=torch.long).T
        vals = torch.ones(len(pairs))
    return torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()


def two_triangle_batch(channels=8, seed=0):
    """Build a single-complex batch duck-typing what TopoTune.forward needs.

    Parameters
    ----------
    channels : int
        Feature dimension of every rank.
    seed : int
        Seed for the random per-rank features.

    Returns
    -------
    types.SimpleNamespace
        Batch with x_0/x_1/x_2, cell_statistics, and one (sparse) matrix
        per supported neighborhood.
    """
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
        {(a, b) for face in _FACES for a in face for b in face if a != b}
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

    return SimpleNamespace(
        x_0=torch.randn(n0, channels, generator=g),
        x_1=torch.randn(n1, channels, generator=g),
        x_2=torch.randn(n2, channels, generator=g),
        cell_statistics=torch.tensor([[n0, n1, n2]], dtype=torch.long),
        **mats,
    )


def build_backbone(neighborhoods, channels=8, layers=1, seed=0):
    """Build a TopoTune backbone over an arbitrary neighborhood list.

    Route GNNs are deepcopies of one seeded template; use
    ``randomize_route_weights_`` to give every route distinct weights.

    Parameters
    ----------
    neighborhoods : list[str]
        Neighborhood list; player/bit order is the order of this list.
    channels : int
        Feature dimension of every rank.
    layers : int
        Number of TopoTune layers.
    seed : int
        Seed for the GNN template initialization.

    Returns
    -------
    TopoTune
        Backbone in eval mode.
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
        neighborhoods=list(neighborhoods),
        layers=layers,
        use_edge_attr=False,
        activation="relu",
    )
    backbone.eval()
    return backbone


def randomize_route_weights_(backbone, seed=0):
    """Give every (layer, route) GNN its own deterministic random weights.

    Distinct per-route weights make alignment between coalition bits and
    routes observable: a bit driving the wrong route changes the output.

    Parameters
    ----------
    backbone : TopoTune
        Backbone to reinitialize in place.
    seed : int
        Base seed; (layer, route) uses ``seed + 1000*layer + route``.
    """
    for layer_idx, layer in enumerate(backbone.graph_routes):
        for route_idx, gnn in enumerate(layer):
            g = torch.Generator().manual_seed(
                seed + 1000 * layer_idx + route_idx
            )
            with torch.no_grad():
                for p in gnn.parameters():
                    p.copy_(0.1 * torch.randn(p.shape, generator=g))
