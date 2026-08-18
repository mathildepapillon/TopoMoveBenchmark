"""Explanation-quality metrics consuming the GraphXAI ground truth.

Conventions match the GraphXAI benchmark suite (Agarwal et al., 2023), so
numbers are comparable to its SubgraphX/GNNExplainer/PGExplainer baselines:

* **GEA** (explanation accuracy): Jaccard agreement between the
  explanation's node set and a ground-truth node mask, maximized over the
  admissible ground-truth candidates (e.g. one per benzene ring, or one
  per matched mutagenic substructure). The datasets in
  :mod:`topobench.data.datasets.graphxai_dataset` carry the union mask on
  every graph (``Data.expl_node_mask``) and the per-graph candidate masks
  in the ``gt_candidates`` sidecar; :func:`graph_explanation_accuracy`
  consumes either directly.
* **GEF** (unfaithfulness): ``1 - exp(-KL(f(graph) || f(graph masked to
  the explanation)))`` over class distributions; lower is better. A pure
  function of the two probability vectors — producing them (two forward
  passes) is the caller's business.

Cell-level attributions (:class:`topobench.explain.cells.CellExplanation`)
are bridged to node-level explanations by :func:`node_projected_scores`:
rank-0 players score their own node, higher-rank players spread their
score onto the nodes of their cell, and each node keeps the maximum score
over the cells that contain it. Top-k'ing those scores to the candidate
size (:func:`top_nodes_by_attribution`) mirrors GraphXAI's protocol.

Truncation caveat: ``GraphXAIDataset`` keeps the full subset lattice of
matched substructures only while it stays within its
``MAX_GT_CANDIDATES`` bound; larger lattices are truncated to the
individual matches plus their full union. An explanation matching a
dropped subset-union candidate then scores below 1 although GraphXAI's
untruncated metric would award 1.0 — truncated lattices can understate
GEA relative to published GraphXAI numbers.
"""

import numpy as np
import torch


def _as_node_set(nodes) -> set[int]:
    """Normalize a node selection to a set of node indices.

    Parameters
    ----------
    nodes : set or collection of int or torch.Tensor or np.ndarray
        Either an explicit collection of node indices, or a boolean /
        0-1 mask of shape ``[num_nodes]`` (the datasets'
        ``expl_node_mask`` convention) whose nonzero positions are the
        selected nodes.

    Returns
    -------
    set of int
        The selected node indices.
    """
    if isinstance(nodes, (torch.Tensor, np.ndarray)):
        tensor = torch.as_tensor(nodes)
        if tensor.dim() != 1:
            raise ValueError(
                f"a node mask must be 1-d [num_nodes], got shape "
                f"{tuple(tensor.shape)}"
            )
        return {int(i) for i in tensor.nonzero(as_tuple=True)[0]}
    return {int(i) for i in nodes}


def gea_jaccard(explanation_nodes, ground_truth_nodes) -> float:
    """Jaccard agreement between an explanation and one ground truth.

    ``1`` means exact recovery; two empty selections also score ``1``
    (there was nothing to find and nothing was claimed).

    Parameters
    ----------
    explanation_nodes : set or torch.Tensor
        Explanation node indices, or a ``[num_nodes]`` mask.
    ground_truth_nodes : set or torch.Tensor
        Ground-truth node indices, or a ``[num_nodes]`` mask.

    Returns
    -------
    float
        ``|intersection| / |union|``.
    """
    pred = _as_node_set(explanation_nodes)
    truth = _as_node_set(ground_truth_nodes)
    if not pred and not truth:
        return 1.0
    return len(pred & truth) / len(pred | truth)


def _candidate_rows(gt_candidates) -> list:
    """Normalize ground-truth candidates to a list of node masks.

    Parameters
    ----------
    gt_candidates : dict or torch.Tensor
        Either one per-graph entry of ``GraphXAIDataset.gt_candidates``
        (a dict whose ``"node"`` key holds ``[K, num_nodes]`` bool
        masks), a ``[K, num_nodes]`` tensor of candidate masks, or a
        single ``[num_nodes]`` mask (e.g. ``Data.expl_node_mask``)
        treated as the only candidate.

    Returns
    -------
    list of torch.Tensor
        One ``[num_nodes]`` mask per admissible candidate.
    """
    if isinstance(gt_candidates, dict):
        gt_candidates = gt_candidates["node"]
    tensor = torch.as_tensor(gt_candidates)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[0] == 0:
        raise ValueError(
            "gt_candidates must be [K, num_nodes] with K >= 1 (or one "
            f"[num_nodes] mask), got shape {tuple(tensor.shape)}"
        )
    return list(tensor)


def graph_explanation_accuracy(explanation_nodes, gt_candidates) -> float:
    """GEA: Jaccard against the best admissible ground-truth candidate.

    GraphXAI provides several admissible explanations per graph and its
    accuracy metric maximizes over them — recovering any one benzene
    ring (or any subset-union of matched substructures) is full credit.
    Passing the single union mask ``Data.expl_node_mask`` instead of the
    ``gt_candidates`` sidecar entry collapses the max to one candidate
    and generally scores lower (see the module docstring, including the
    ``MAX_GT_CANDIDATES`` truncation caveat).

    Parameters
    ----------
    explanation_nodes : set or torch.Tensor
        Explanation node indices, or a ``[num_nodes]`` mask.
    gt_candidates : dict or torch.Tensor
        Admissible ground-truth candidates; see :func:`_candidate_rows`
        for the accepted forms.

    Returns
    -------
    float
        ``max_c gea_jaccard(explanation, candidate_c)``.
    """
    pred = _as_node_set(explanation_nodes)
    return max(
        gea_jaccard(pred, row) for row in _candidate_rows(gt_candidates)
    )


def node_projected_scores(
    phi, players, num_nodes: int, cell_nodes=None
) -> np.ndarray:
    """Project a cell-level attribution to per-node scores.

    Rank-0 players score their own node (``player.index``). Higher-rank
    players spread their score onto the nodes of their cell, given by
    ``cell_nodes``; each node keeps the **maximum** score over the cells
    that contain it, so a node inside a strongly attributed face ranks
    at least as high as that face. Nodes covered by no player score
    ``-inf``, keeping them out of any top-k.

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player, in player (bit) order.
    players : list of CellPlayer
        The explained cells, in player (bit) order.
    num_nodes : int
        Number of rank-0 cells in the graph.
    cell_nodes : callable, optional
        Maps a player with ``rank > 0`` to the iterable of node indices
        of its cell (e.g. from the lifting's incidence). Required as
        soon as a non-rank-0 player is present.

    Returns
    -------
    np.ndarray
        Score per node, shape ``[num_nodes]``.
    """
    scores = np.full(num_nodes, -np.inf)
    for value, player in zip(phi, players, strict=True):
        if player.rank == 0:
            nodes = [player.index]
        elif cell_nodes is None:
            raise ValueError(
                f"player {player} has rank {player.rank} > 0: projecting "
                "it to nodes needs cell_nodes (player -> node indices)"
            )
        else:
            nodes = list(cell_nodes(player))
        for node in nodes:
            scores[node] = max(scores[node], float(value))
    return scores


def top_nodes_by_attribution(node_scores, k: int) -> set[int]:
    """The ``k`` highest-scored nodes, as a set (GraphXAI's top-k'ing).

    GraphXAI evaluates explainers at the ground-truth size: the
    explanation is the top-``k`` nodes where ``k`` is the candidate's
    size. Ties are broken by node index (stable sort), matching the
    main-repo implementation.

    Parameters
    ----------
    node_scores : np.ndarray
        Score per node.
    k : int
        Number of nodes to keep.

    Returns
    -------
    set of int
        Indices of the ``k`` highest-scored nodes.
    """
    order = np.argsort(-np.asarray(node_scores), kind="stable")[:k]
    return {int(i) for i in order}


def gef_unfaithfulness(p_full, p_masked) -> float:
    """GEF: ``1 - exp(-KL(p_full || p_masked))`` over class distributions.

    Both inputs are probability vectors (softmax outputs): ``p_full``
    from the unmasked graph, ``p_masked`` from the graph masked to the
    explanation. ``0`` is perfectly faithful, ``1`` maximally
    unfaithful; lower is better.

    Parameters
    ----------
    p_full : np.ndarray
        Class probabilities of the model on the full input.
    p_masked : np.ndarray
        Class probabilities of the model on the explanation-masked
        input.

    Returns
    -------
    float
        The unfaithfulness score in ``[0, 1)``.
    """
    p = np.clip(np.asarray(p_full, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(p_masked, dtype=float), 1e-12, 1.0)
    p, q = p / p.sum(), q / q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return 1.0 - float(np.exp(-kl))


def random_control_gea(
    num_nodes: int,
    gt_candidates,
    k: int,
    n_draws: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean and sd of GEA under uniformly random ``k``-subsets of nodes.

    The floor every explainer must beat for its GEA to mean anything:
    on small graphs with large motifs, random selections already score
    substantially above zero.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to draw from.
    gt_candidates : dict or torch.Tensor
        Admissible ground-truth candidates; see :func:`_candidate_rows`.
    k : int
        Size of each random selection.
    n_draws : int
        Number of random selections.
    seed : int
        Seed for the draws.

    Returns
    -------
    tuple of float
        ``(mean, sd)`` of :func:`graph_explanation_accuracy` over the
        draws.
    """
    rng = np.random.default_rng(seed)
    rows = _candidate_rows(gt_candidates)
    scores = [
        graph_explanation_accuracy(
            {int(i) for i in rng.choice(num_nodes, size=k, replace=False)},
            torch.stack(rows),
        )
        for _ in range(n_draws)
    ]
    return float(np.mean(scores)), float(np.std(scores))
