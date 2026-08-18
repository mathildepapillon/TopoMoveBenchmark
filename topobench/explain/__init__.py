"""Shapley-based explainability for topological models.

This subpackage treats explanation as a cooperative game: players are
components of a model or its input, coalitions are encoded as integer
bitmasks (bit ``i`` = player ``i``), and a game ``v(mask) -> float`` scores
each coalition. Two uses are supported:

* **Explain a prediction** (cells as players): :func:`explain_cells` masks
  the features of absent cells and attributes one model output to the cells
  of the complex — exact Shapley values for small player sets, permutation
  sampling beyond that. Models that consume per-hop encodings (HOPSE) need
  :class:`HopseCellMaskingGame`, which masks at the interface they actually
  read.
* **Explain performance** (neighborhoods as players): wrap a TopoTune
  backbone in :class:`CoalitionMaskedBackbone`, score coalitions of
  message-passing neighborhoods with any metric, attribute it with
  :func:`sampled_shapley`, select a coalition (top-k, anchored, greedy, or
  an automatic-size rule), then :func:`prune_backbone_` and keep training.

For end-to-end neighborhood selection, :func:`run_ladder` packages the
performance game into one fixed pipeline: all-players stem ->
backward-elimination ladder on the masked validation game -> train B
rungs -> validation selects. Every attribution can be audited with the
executable axiom checks in :mod:`topobench.explain.axioms`, and on the
GraphXAI datasets explanations can be scored against the shipped ground
truth with the metrics in :mod:`topobench.explain.metrics` (GEA
explanation accuracy, GEF unfaithfulness).
"""

from .axioms import (
    check_batch_invariance,
    check_efficiency,
    check_symmetry_pair,
    find_null_players,
)
from .cells import (
    DEFAULT_MAX_EXACT_EVALUATIONS,
    EXACT_PLAYER_LIMIT,
    CellExplanation,
    CellMaskingGame,
    CellPlayer,
    HopseCellMaskingGame,
    explain_cells,
)
from .games import CachedGame, Game, TabulatedGame
from .ladder import (
    LadderResult,
    LadderRung,
    backward_elimination_ladder,
    binomial_se,
    run_ladder,
)
from .metrics import (
    gea_jaccard,
    gef_unfaithfulness,
    graph_explanation_accuracy,
    node_projected_scores,
    random_control_gea,
    top_nodes_by_attribution,
)
from .neighborhoods import (
    CoalitionMaskedBackbone,
    coalition_to_mask,
    mask_to_coalition,
    prune_backbone_,
)
from .selection import (
    SubmodularityCensus,
    anchored_top_k_mask,
    cost_regularized,
    greedy_mask,
    greedy_with_noise_stop,
    smallest_sufficient_coalition,
    submodularity_census,
    threshold_at_zero,
    top_k_mask,
)
from .shapley import (
    SampledAttribution,
    efficiency_gap,
    resample_pick_stability,
    sampled_shapley,
    shapley_interaction,
    shapley_values,
)

__all__ = [
    "DEFAULT_MAX_EXACT_EVALUATIONS",
    "EXACT_PLAYER_LIMIT",
    "CachedGame",
    "CellExplanation",
    "CellMaskingGame",
    "CellPlayer",
    "CoalitionMaskedBackbone",
    "Game",
    "HopseCellMaskingGame",
    "LadderResult",
    "LadderRung",
    "SampledAttribution",
    "SubmodularityCensus",
    "TabulatedGame",
    "anchored_top_k_mask",
    "backward_elimination_ladder",
    "binomial_se",
    "check_batch_invariance",
    "check_efficiency",
    "check_symmetry_pair",
    "coalition_to_mask",
    "cost_regularized",
    "efficiency_gap",
    "explain_cells",
    "find_null_players",
    "gea_jaccard",
    "gef_unfaithfulness",
    "graph_explanation_accuracy",
    "greedy_mask",
    "greedy_with_noise_stop",
    "mask_to_coalition",
    "node_projected_scores",
    "prune_backbone_",
    "random_control_gea",
    "resample_pick_stability",
    "run_ladder",
    "sampled_shapley",
    "shapley_interaction",
    "shapley_values",
    "smallest_sufficient_coalition",
    "submodularity_census",
    "threshold_at_zero",
    "top_k_mask",
    "top_nodes_by_attribution",
]
