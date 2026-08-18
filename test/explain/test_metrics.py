"""Hand-computable tests of the GraphXAI explanation-quality metrics.

Every asserted value is derivable on paper from small synthetic graphs
with known ground-truth candidates, including the max-over-candidates
behavior and its interaction with the dataset's MAX_GT_CANDIDATES
truncation (which can understate GEA vs published GraphXAI numbers).
"""

import numpy as np
import pytest
import torch

from topobench.data.datasets.graphxai_dataset import (
    MAX_GT_CANDIDATES,
    GraphXAIDataset,
)
from topobench.explain import (
    CellPlayer,
    gea_jaccard,
    gef_unfaithfulness,
    graph_explanation_accuracy,
    node_projected_scores,
    random_control_gea,
    top_nodes_by_attribution,
)


def mask(num_nodes, nodes):
    """Boolean node mask with the given indices set.

    Parameters
    ----------
    num_nodes : int
        Mask length.
    nodes : iterable of int
        Indices to set.

    Returns
    -------
    torch.Tensor
        Bool mask of shape ``[num_nodes]``.
    """
    m = torch.zeros(num_nodes, dtype=torch.bool)
    m[list(nodes)] = True
    return m


# ------------------------------------------------------------ gea_jaccard


def test_gea_jaccard_hand_values():
    """|intersection| / |union| on sets and on masks, by hand."""
    assert gea_jaccard({0, 1}, {1, 2}) == pytest.approx(1 / 3)
    assert gea_jaccard({0, 1, 2}, {0, 1, 2}) == 1.0
    assert gea_jaccard({0, 1}, {2, 3}) == 0.0
    # masks and sets are interchangeable
    assert gea_jaccard(mask(4, [0, 1]), {1, 2}) == pytest.approx(1 / 3)
    assert gea_jaccard({0, 1}, mask(4, [1, 2]).float()) == pytest.approx(
        1 / 3
    )


def test_gea_jaccard_empty_vs_empty_is_one():
    """Nothing to find and nothing claimed is exact recovery."""
    assert gea_jaccard(set(), set()) == 1.0
    assert gea_jaccard(set(), mask(3, [])) == 1.0
    assert gea_jaccard({0}, set()) == 0.0


def test_node_mask_must_be_one_dimensional():
    """A [K, num_nodes] tensor is a candidate stack, not one mask."""
    with pytest.raises(ValueError, match="1-d"):
        gea_jaccard(torch.zeros(2, 3, dtype=torch.bool), {0})


# ---------------------------------------------------- max over candidates


def test_gea_maximizes_over_admissible_candidates():
    """Recovering ANY one candidate is full credit; the union is not.

    Two disjoint 3-node motifs on 8 nodes. An explanation equal to the
    first motif has Jaccard 1 with candidate 0 and 0 with candidate 1;
    the max makes GEA 1. Scoring against the union mask instead (what
    Data.expl_node_mask stores) gives only 3/6.
    """
    candidates = torch.stack([mask(8, [0, 1, 2]), mask(8, [3, 4, 5])])
    pred = {0, 1, 2}
    assert graph_explanation_accuracy(pred, candidates) == 1.0
    union = candidates.any(0)
    assert graph_explanation_accuracy(pred, union) == pytest.approx(0.5)


def test_gea_hand_value_on_a_partial_overlap():
    """Max picks the better partial overlap: 2/4 beats 1/5."""
    candidates = torch.stack([mask(8, [0, 1, 2]), mask(8, [3, 4, 5])])
    pred = {2, 3, 4}
    # candidate 0: inter {2}, union {0,1,2,3,4} -> 1/5
    # candidate 1: inter {3,4}, union {2,3,4,5} -> 2/4
    assert graph_explanation_accuracy(pred, candidates) == pytest.approx(
        0.5
    )


def test_gea_accepts_the_dataset_sidecar_dict():
    """One entry of GraphXAIDataset.gt_candidates works directly."""
    entry = {
        "node": torch.stack([mask(6, [0, 1]), mask(6, [4, 5])]),
        "edge": torch.zeros(2, 0, dtype=torch.bool),
    }
    assert graph_explanation_accuracy({4, 5}, entry) == 1.0
    assert graph_explanation_accuracy({0, 5}, entry) == pytest.approx(
        1 / 3
    )


def test_gea_refuses_an_empty_candidate_stack():
    """Zero candidates is a caller bug, not a 0.0 score."""
    with pytest.raises(ValueError, match="K >= 1"):
        graph_explanation_accuracy(
            {0}, torch.zeros(0, 4, dtype=torch.bool)
        )


# ----------------------------------- MAX_GT_CANDIDATES truncation interplay


def disjoint_singles(n_matches, nodes_per_match):
    """Disjoint substructure matches: match i covers a distinct node block.

    Parameters
    ----------
    n_matches : int
        Number of matched substructures.
    nodes_per_match : int
        Nodes per match; the graph has ``n_matches * nodes_per_match``
        nodes.

    Returns
    -------
    tuple
        (node mask list, edge mask list) as ``_candidate_lattice``
        expects.
    """
    num_nodes = n_matches * nodes_per_match
    singles_n = [
        mask(
            num_nodes,
            range(i * nodes_per_match, (i + 1) * nodes_per_match),
        )
        for i in range(n_matches)
    ]
    singles_e = [torch.zeros(0, dtype=torch.bool) for _ in singles_n]
    return singles_n, singles_e


def test_full_lattice_awards_subset_unions_full_credit():
    """With few matches the lattice holds every subset union: GEA 1.0.

    3 disjoint 2-node matches -> 2^3 - 1 = 7 <= MAX_GT_CANDIDATES
    candidates, including the union of matches 0 and 1. An explanation
    equal to exactly that union scores 1.0.
    """
    singles_n, singles_e = disjoint_singles(3, 2)
    cand_n, _ = GraphXAIDataset._candidate_lattice(singles_n, singles_e)
    assert cand_n.shape[0] == 7
    pred = {0, 1, 2, 3}  # matches 0 and 1
    assert graph_explanation_accuracy(pred, cand_n) == 1.0


def test_truncated_lattice_understates_gea():
    """Past the bound only singles + full union survive: GEA drops.

    6 disjoint 2-node matches -> 2^6 - 1 = 63 > MAX_GT_CANDIDATES, so
    the lattice is truncated to the 6 singles plus their union (7
    candidates). The same two-match explanation now scores only
    max(2/4 against a single, 4/12 against the union) = 0.5 — the
    truncation understates GEA relative to the untruncated GraphXAI
    metric, which would award 1.0 (this is why fork numbers can sit
    below published GraphXAI numbers on many-match graphs).
    """
    assert 2**6 - 1 > MAX_GT_CANDIDATES
    singles_n, singles_e = disjoint_singles(6, 2)
    cand_n, _ = GraphXAIDataset._candidate_lattice(singles_n, singles_e)
    assert cand_n.shape[0] == 7  # 6 singles + full union
    pred = {0, 1, 2, 3}  # matches 0 and 1: in the dropped part of the lattice
    assert graph_explanation_accuracy(pred, cand_n) == pytest.approx(0.5)


# --------------------------------------------------------- node projection


def test_node_projected_scores_and_top_k():
    """Rank-0 players score their node; a face spreads to its nodes.

    phi: node0=0.1, node1=0.9, face{2,3}=0.5. Projection: [0.1, 0.9,
    0.5, 0.5]; top-2 = {1, 2} (index tie-break between the two face
    nodes picks node 2).
    """
    players = [
        CellPlayer(rank=0, index=0),
        CellPlayer(rank=0, index=1),
        CellPlayer(rank=2, index=0),
    ]
    phi = np.array([0.1, 0.9, 0.5])
    scores = node_projected_scores(
        phi, players, num_nodes=4, cell_nodes=lambda p: {2, 3}
    )
    assert np.allclose(scores, [0.1, 0.9, 0.5, 0.5])
    assert top_nodes_by_attribution(scores, 2) == {1, 2}
    assert top_nodes_by_attribution(scores, 3) == {1, 2, 3}


def test_node_projection_keeps_the_maximum_over_covering_cells():
    """A node inside a strong face outranks its own weak attribution."""
    players = [
        CellPlayer(rank=0, index=0),
        CellPlayer(rank=2, index=0),
    ]
    phi = np.array([0.1, 0.8])
    scores = node_projected_scores(
        phi, players, num_nodes=2, cell_nodes=lambda p: {0, 1}
    )
    assert np.allclose(scores, [0.8, 0.8])


def test_node_projection_requires_cell_nodes_for_higher_ranks():
    """Higher-rank players without a projection must fail loudly."""
    players = [CellPlayer(rank=1, index=0)]
    with pytest.raises(ValueError, match="cell_nodes"):
        node_projected_scores(np.array([1.0]), players, num_nodes=2)


def test_uncovered_nodes_never_enter_the_top_k():
    """Nodes covered by no player score -inf and are picked last."""
    players = [CellPlayer(rank=0, index=2)]
    scores = node_projected_scores(np.array([-5.0]), players, num_nodes=3)
    assert top_nodes_by_attribution(scores, 1) == {2}


# ------------------------------------------------------------------- GEF


def test_gef_is_zero_when_masking_changes_nothing():
    """Identical distributions have zero KL, hence zero unfaithfulness."""
    p = np.array([0.3, 0.7])
    assert gef_unfaithfulness(p, p) == pytest.approx(0.0, abs=1e-12)


def test_gef_hand_value():
    """KL([.5,.5] || [.25,.75]) = 0.5 ln(4/3); GEF = 1 - exp(-KL)."""
    kl = 0.5 * np.log(4 / 3)
    expected = 1.0 - np.exp(-kl)
    assert gef_unfaithfulness(
        [0.5, 0.5], [0.25, 0.75]
    ) == pytest.approx(expected)


def test_gef_increases_with_divergence():
    """A masked distribution further from p_full is less faithful."""
    p = [0.5, 0.5]
    assert gef_unfaithfulness(p, [0.4, 0.6]) < gef_unfaithfulness(
        p, [0.1, 0.9]
    )


# ------------------------------------------------------------ random floor


def test_random_control_gea_floor_is_hand_computable():
    """With k = num_nodes every draw is the full node set: the floor is
    exactly Jaccard(all nodes, best candidate), with zero variance."""
    candidates = torch.stack([mask(4, [0, 1]), mask(4, [3])])
    mean, sd = random_control_gea(4, candidates, k=4, n_draws=10)
    assert mean == pytest.approx(0.5)  # {0,1,2,3} vs {0,1} -> 2/4
    assert sd == pytest.approx(0.0)


def test_random_control_gea_is_a_floor_for_a_perfect_explainer():
    """A perfect explanation beats the random floor on a sparse motif."""
    candidates = torch.stack([mask(10, [0, 1])])
    mean, _ = random_control_gea(10, candidates, k=2, n_draws=200, seed=1)
    perfect = graph_explanation_accuracy({0, 1}, candidates)
    assert perfect == 1.0
    assert mean < perfect
