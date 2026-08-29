"""GraphXAI explanation metrics on cell players (GEA / GEF).

Verbatim port of the validated exp #1 module
(data/frozen/reference/exp1_worktree/topobench_repo/topobench/explain/
metrics.py). GEA maximizes the Jaccard overlap at ground-truth size over
GraphXAI's admissible candidate explanations, after lifting node/edge
masks to cells (2-cell rule: "all" boundary edges marked, or "majority").
"""

from __future__ import annotations

import numpy as np


def lift_masks_to_cells(
    data,
    node_mask: np.ndarray,
    edge_mask: np.ndarray,
    offset: dict,
    n_cells: dict,
    ring_rule: str = "all",
) -> np.ndarray:
    """Lift a node/edge ground-truth mask to a cell mask over game players.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The lifted complex; must carry ``cell_1_endpoints``, ``cell_2_boundary_flat`` and
        ``cell_2_boundary_ptr`` (see ``CellCycleLiftingGT``) plus the original
        ``edge_index``.
    node_mask : numpy.ndarray
        Boolean mask over original nodes.
    edge_mask : numpy.ndarray
        Boolean mask over original (directed) edges.
    offset : dict
        Rank to global player offset.
    n_cells : dict
        Rank to number of cells.
    ring_rule : str
        ``"all"`` marks a 2-cell when every boundary 1-cell is marked; ``"majority"``
        when more than half are.

    Returns
    -------
    numpy.ndarray
        Boolean mask over game players.
    """
    total = sum(n_cells.values())
    out = np.zeros(total, dtype=bool)

    node_mask = np.asarray(node_mask, dtype=bool)
    edge_mask = np.asarray(edge_mask, dtype=bool)

    # rank 0: 0-cell order is the node order (asserted by the lifting)
    if 0 in n_cells:
        out[offset[0] : offset[0] + n_cells[0]] = node_mask[: n_cells[0]]

    # rank 1: match each 1-cell to the original undirected edge
    edge_lookup: dict[tuple[int, int], bool] = {}
    edge_index = data.edge_index.cpu().numpy()
    for pos in range(edge_index.shape[1]):
        u, v = int(edge_index[0, pos]), int(edge_index[1, pos])
        key = (u, v) if u <= v else (v, u)
        edge_lookup[key] = edge_lookup.get(key, False) or bool(edge_mask[pos])

    endpoints = data.cell_1_endpoints.cpu().numpy()
    cell1 = np.zeros(n_cells.get(1, 0), dtype=bool)
    for j in range(cell1.shape[0]):
        u, v = int(endpoints[j, 0]), int(endpoints[j, 1])
        key = (u, v) if u <= v else (v, u)
        cell1[j] = edge_lookup.get(key, False)
    if 1 in n_cells:
        out[offset[1] : offset[1] + n_cells[1]] = cell1

    # rank 2: a ring is important when its boundary edges are
    if 2 in n_cells and n_cells[2] > 0:
        flat = data.cell_2_boundary_flat.cpu().numpy()
        ptr = data.cell_2_boundary_ptr.cpu().numpy()
        cell2 = np.zeros(n_cells[2], dtype=bool)
        for j in range(n_cells[2]):
            boundary = flat[ptr[j] : ptr[j + 1]]
            if boundary.size == 0:
                continue
            marked = cell1[boundary]
            cell2[j] = (
                bool(marked.all())
                if ring_rule == "all"
                else bool(marked.mean() > 0.5)
            )
        out[offset[2] : offset[2] + n_cells[2]] = cell2
    return out


def candidate_cell_masks(
    data, candidates: dict, offset: dict, n_cells: dict, ring_rule: str = "all"
) -> np.ndarray:
    """Lift every candidate ground-truth explanation to cells.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The lifted complex.
    candidates : dict
        ``{"node": [K, num_nodes], "edge": [K, num_edges]}`` boolean tensors.
    offset : dict
        Rank to global player offset.
    n_cells : dict
        Rank to number of cells.
    ring_rule : str
        Ring rule, see :func:`lift_masks_to_cells`.

    Returns
    -------
    numpy.ndarray
        Boolean array ``[K, n_players]``.
    """
    node = candidates["node"].cpu().numpy()
    edge = candidates["edge"].cpu().numpy()
    return np.stack(
        [
            lift_masks_to_cells(
                data, node[i], edge[i], offset, n_cells, ring_rule
            )
            for i in range(node.shape[0])
        ]
    )


def top_k_indices(attribution: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest attributions.

    Parameters
    ----------
    attribution : numpy.ndarray
        Attribution per player.
    k : int
        Number of players to select.

    Returns
    -------
    numpy.ndarray
        Selected indices.
    """
    k = int(min(max(k, 0), attribution.shape[0]))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-attribution, kind="stable")
    return order[:k]


def graph_explanation_accuracy(
    attribution: np.ndarray, gt_masks: np.ndarray
) -> dict:
    """GEA: max over candidates of the Jaccard overlap at the ground-truth count.

    Parameters
    ----------
    attribution : numpy.ndarray
        Attribution per player (larger is more important).
    gt_masks : numpy.ndarray
        Boolean array ``[K, n_players]`` of candidate ground-truth cell masks.

    Returns
    -------
    dict
        Best Jaccard, the candidate achieving it, and its ground-truth size.
    """
    best = {"gea": float("nan"), "candidate": -1, "gt_size": 0}
    for c in range(gt_masks.shape[0]):
        gt = gt_masks[c]
        size = int(gt.sum())
        if size == 0:
            continue
        selected = np.zeros_like(gt)
        selected[top_k_indices(attribution, size)] = True
        intersection = int((selected & gt).sum())
        union = int((selected | gt).sum())
        jaccard = intersection / union if union else float("nan")
        if np.isnan(best["gea"]) or jaccard > best["gea"]:
            best = {"gea": float(jaccard), "candidate": int(c), "gt_size": size}
    return best


def graph_explanation_faithfulness(
    game, attribution: np.ndarray, top_fraction: float = 0.25
) -> dict:
    """GEF: unfaithfulness of the top-fraction coalition, ``1 - exp(-KL)``.

    The KL is taken between the softmax of the model's logits on the coalition and on the
    full complex, in the same direction as GraphXAI's implementation.

    Parameters
    ----------
    game : MaskedGCCNGame
        The game (supplies the masked forward pass and the full-complex logits).
    attribution : numpy.ndarray
        Attribution per player.
    top_fraction : float
        Fraction of players kept.

    Returns
    -------
    dict
        GEF, the KL value and the coalition size.
    """
    n = attribution.shape[0]
    k = int(np.ceil(top_fraction * n))
    coalition = np.zeros((1, n), dtype=bool)
    coalition[0, top_k_indices(attribution, k)] = True
    pert_logits = game.logits(coalition)[0]

    def softmax(z):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    p_orig = softmax(game.model_logits_full)
    p_pert = softmax(pert_logits)
    eps = 1e-12
    kl = float(
        np.sum(p_pert * (np.log(p_pert + eps) - np.log(p_orig + eps)))
    )
    return {
        "gef": float(1.0 - np.exp(-kl)),
        "gef_kl": kl,
        "gef_coalition_size": k,
    }


def cells_to_nodes(cell_indices, data, offset: dict, n_cells: dict) -> set:
    """Project a set of cell players to the original node set.

    A 0-cell contributes its node; a 1-cell its two endpoints; a 2-cell
    every endpoint of its boundary 1-cells. Reconstruction of exp #4's
    node projection (its code was not ingested); the rule is stated in
    the paper wherever node-projected GEA is reported.
    """
    endpoints = data.cell_1_endpoints.cpu().numpy()
    flat = data.cell_2_boundary_flat.cpu().numpy()
    ptr = data.cell_2_boundary_ptr.cpu().numpy()
    nodes: set[int] = set()
    for g in cell_indices:
        g = int(g)
        if 0 in n_cells and offset[0] <= g < offset[0] + n_cells[0]:
            nodes.add(g - offset[0])
        elif 1 in n_cells and offset[1] <= g < offset[1] + n_cells[1]:
            j = g - offset[1]
            nodes.update(int(v) for v in endpoints[j])
        elif 2 in n_cells and offset.get(2, -1) <= g:
            j = g - offset[2]
            for e in flat[ptr[j]:ptr[j + 1]]:
                nodes.update(int(v) for v in endpoints[int(e)])
    return nodes


def node_projected_gea(attribution, data, candidates, offset: dict,
                       n_cells: dict, gt_masks_cells) -> dict:
    """Node-level GEA: project the top-k cells to nodes, Jaccard against
    each candidate's ground-truth node mask, max over candidates.

    ``k`` per candidate is that candidate's cell-level ground-truth
    count (same budget as the cell-level GEA).
    """
    import numpy as _np

    node_masks = candidates["node"].cpu().numpy().astype(bool)
    best = {"gea_node": float("nan"), "candidate": -1}
    for c in range(node_masks.shape[0]):
        k = int(gt_masks_cells[c].sum())
        if k == 0:
            continue
        sel_cells = top_k_indices(_np.asarray(attribution), k)
        sel_nodes = cells_to_nodes(sel_cells, data, offset, n_cells)
        gt_nodes = set(_np.where(node_masks[c])[0].tolist())
        if not sel_nodes and not gt_nodes:
            j = 1.0
        else:
            j = len(sel_nodes & gt_nodes) / len(sel_nodes | gt_nodes)
        if _np.isnan(best["gea_node"]) or j > best["gea_node"]:
            best = {"gea_node": float(j), "candidate": int(c)}
    return best
