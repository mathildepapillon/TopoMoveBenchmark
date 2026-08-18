"""Vendored GraphXAI helpers (MIT License, Copyright (c) 2023 GraphXAI).

Only the pieces needed to reproduce the ground-truth explanation masks of the four
GraphXAI molecular datasets are vendored here; see ``substruct_chem_match.py`` and
``gxai_utils.py``.
"""

from .gxai_utils import (
    edge_mask_from_node_mask,
    match_edge_presence,
    to_networkx_conv,
)

__all__ = [
    "edge_mask_from_node_mask",
    "match_edge_presence",
    "to_networkx_conv",
]
