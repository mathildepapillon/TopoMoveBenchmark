"""Vendored helpers from GraphXAI (mims-harvard/GraphXAI), MIT License, (c) 2023 GraphXAI.

Adapted from ``graphxai/utils/misc.py`` and ``graphxai/utils/nx_conversion.py`` at
upstream commit a11e65ffbc4df737f35522a8accf2283a8aeaa37. The
functions are kept semantically identical to the originals, because the ground-truth
explanation masks of the GraphXAI datasets are *defined* by them: an edge is part of an
explanation exactly when the original code says it is. Only vectorization and docstring
style were changed.
"""

import networkx as nx
import torch


def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    """Convert a node mask to an edge mask (an edge counts when both ends are in the mask).

    Parameters
    ----------
    node_mask : torch.Tensor
        Boolean mask over all nodes indexed by ``edge_index``.
    edge_index : torch.Tensor
        Edge index of shape ``[2, num_edges]``.

    Returns
    -------
    torch.Tensor
        Float mask of shape ``[num_edges]``.
    """
    keep = node_mask.bool()
    return (keep[edge_index[0]] & keep[edge_index[1]]).float()


def match_edge_presence(edge_index: torch.Tensor, node_idx):
    """Return an edge mask marking every edge incident to any node in ``node_idx``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index of shape ``[2, num_edges]``.
    node_idx : torch.Tensor or int or list
        Node index/indices.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``[num_edges]``.
    """
    nodes = torch.as_tensor(node_idx).flatten()
    emask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for ni in nodes:
        emask = emask | (edge_index[0, :] == ni) | (edge_index[1, :] == ni)
    return emask


def to_networkx_conv(
    data,
    node_attrs=None,
    edge_attrs=None,
    to_undirected: bool = False,
    remove_self_loops: bool = False,
    get_map: bool = False,
):
    """Convert a PyG ``Data`` object to a networkx graph, GraphXAI-style.

    Unlike ``torch_geometric.utils.to_networkx``, the node set is taken from the unique
    entries of ``edge_index`` and relabelled contiguously, which is what the vendored
    substructure matchers expect.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph.
    node_attrs : list of str, optional
        Node attributes to copy onto the networkx nodes.
    edge_attrs : list of str, optional
        Edge attributes to copy onto the networkx edges.
    to_undirected : bool
        Whether to build an undirected graph.
    remove_self_loops : bool
        Whether to drop self loops.
    get_map : bool
        If True, also return the original-to-contiguous node index map (and skip
        relabelling back).

    Returns
    -------
    networkx.Graph or tuple
        The converted graph, or ``(graph, map)`` when ``get_map`` is True.
    """
    G = nx.Graph() if to_undirected else nx.DiGraph()

    node_list = sorted(torch.unique(data.edge_index).tolist())
    map_norm = {node_list[i]: i for i in range(len(node_list))}
    rev_map_norm = {v: k for k, v in map_norm.items()}
    G.add_nodes_from([map_norm[n] for n in node_list])

    values = {}
    for key, item in data:
        values[key] = item.squeeze().tolist() if torch.is_tensor(item) else item
        if isinstance(values[key], list | tuple) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        u, v = map_norm[u], map_norm[v]
        if to_undirected and v > u:
            continue
        if remove_self_loops and u == v:
            continue
        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    if get_map:
        return G, map_norm
    return nx.relabel_nodes(G, mapping=rev_map_norm)
