"""TopoSHAP: cell-native exact Shapley-interaction explanations for TDL models.

Two instantiations of one framework:

* Explaining predictions (leg 1): exact Shapley / Shapley-interaction values for
  the cells of a combinatorial complex under a masking game
  (:mod:`toposhap.cells`).
* Explaining model performance (leg 2): the 11 candidate neighborhoods of a
  GCCN treated as players in a cooperative game, driving one-run architecture
  selection (:mod:`toposhap.neighborhoods`, :mod:`toposhap.selection`).

The Shapley engine (:mod:`toposhap.shapley`) is shared and torch-free; only the
model-facing games require the pinned TopoBench checkout under
``external/TopoBench``.
"""

from toposhap.vocabulary import (
    MENU,
    NEIGHBORHOODS,
    coalition_to_mask,
    mask_to_coalition,
)

__all__ = [
    "MENU",
    "NEIGHBORHOODS",
    "coalition_to_mask",
    "mask_to_coalition",
]

__version__ = "0.1.0"
