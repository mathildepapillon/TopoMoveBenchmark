"""Shapley-based explainability for topological models.

This subpackage treats explanation as a cooperative game: players are
components of a model or its input, coalitions are encoded as integer
bitmasks (bit ``i`` = player ``i``), and a game ``v(mask) -> float`` scores
each coalition. Two uses are supported:

* **Explain a prediction** (cells as players): :func:`explain_cells` masks
  the features of absent cells and attributes one model output to the cells
  of the complex — exact Shapley values for small player sets, permutation
  sampling beyond that.
* **Explain performance** (neighborhoods as players): wrap a TopoTune
  backbone in :class:`CoalitionMaskedBackbone`, score coalitions of
  message-passing neighborhoods with any metric, attribute it with
  :func:`sampled_shapley`, select a coalition (top-k, anchored, greedy, or
  an automatic-size rule), then :func:`prune_backbone_` and keep training.

Every attribution can be audited with the executable axiom checks in
:mod:`topobench.explain.axioms`.
"""

from .axioms import (
    check_batch_invariance,
    check_efficiency,
    check_symmetry_pair,
    find_null_players,
)
from .cells import (
    EXACT_PLAYER_LIMIT,
    CellExplanation,
    CellMaskingGame,
    CellPlayer,
    explain_cells,
)
from .games import CachedGame, Game, TabulatedGame
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
    "EXACT_PLAYER_LIMIT",
    "CachedGame",
    "CellExplanation",
    "CellMaskingGame",
    "CellPlayer",
    "CoalitionMaskedBackbone",
    "Game",
    "SampledAttribution",
    "SubmodularityCensus",
    "TabulatedGame",
    "anchored_top_k_mask",
    "check_batch_invariance",
    "check_efficiency",
    "check_symmetry_pair",
    "coalition_to_mask",
    "cost_regularized",
    "efficiency_gap",
    "explain_cells",
    "find_null_players",
    "greedy_mask",
    "greedy_with_noise_stop",
    "mask_to_coalition",
    "prune_backbone_",
    "resample_pick_stability",
    "sampled_shapley",
    "shapley_interaction",
    "shapley_values",
    "smallest_sufficient_coalition",
    "submodularity_census",
    "threshold_at_zero",
    "top_k_mask",
]
