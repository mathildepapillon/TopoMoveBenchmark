"""Coalition masking for TopoTune/GCCN backbones.

The neighborhood game's value function needs the model's output under an
arbitrary coalition of neighborhoods. The masking point is
``aggregate_inter_nbhd`` (gccn.py): per layer, each route's GNN output is
summed into its destination rank — dropping a neighborhood means dropping its
route's contribution from that sum, exactly what "the player is absent" means
for a message-passing route. (For HOPSE, the analogous mechanism is
encoding-block zeroing, validated in #13 by an exhaustive 255-coalition
column-slicing check.)

Two uses:

* :class:`CoalitionMaskedBackbone` — inference-time wrapper for attribution
  games on a trained all-neighborhood stem.
* :func:`prune_backbone_` — permanent in-place prune for the
  prune-and-continue recipe (routes outside the coalition are removed so
  continued training pays only for kept neighborhoods; warm-starting from stem
  weights is free — warm-cold = +0.0024 over 105 subsets, experiment #11).
"""

from __future__ import annotations

import contextlib

import torch

from toposhap.vocabulary import NEIGHBORHOODS, mask_to_coalition


class CoalitionMaskedBackbone(torch.nn.Module):
    """Wraps a TopoTune backbone; forward() respects ``self.coalition_mask``.

    The wrapper monkey-wraps ``aggregate_inter_nbhd`` to zero the contribution
    of routes whose neighborhood bit is absent from the active coalition. The
    backbone must have been built with the full vocabulary in bit order, i.e.
    ``backbone.neighborhoods == list(NEIGHBORHOODS)`` — enforced at init so a
    reordered config can never silently misalign players with routes.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        if list(backbone.neighborhoods) != list(NEIGHBORHOODS):
            raise ValueError(
                "backbone neighborhoods must equal the canonical vocabulary "
                "in bit order; got "
                f"{list(backbone.neighborhoods)}"
            )
        self.backbone = backbone
        self.coalition_mask: int = (1 << len(NEIGHBORHOODS)) - 1
        self._original_aggregate = backbone.aggregate_inter_nbhd
        backbone.aggregate_inter_nbhd = self._masked_aggregate

    def _masked_aggregate(self, x_out_per_route: dict) -> dict:
        kept = {
            route_index: x
            for route_index, x in x_out_per_route.items()
            if self.coalition_mask >> route_index & 1
        }
        x_out_per_rank: dict[int, torch.Tensor] = {}
        for route_index, (_, dst_rank) in enumerate(self.backbone.routes):
            if route_index not in kept:
                continue
            if dst_rank not in x_out_per_rank:
                x_out_per_rank[dst_rank] = kept[route_index]
            else:
                x_out_per_rank[dst_rank] = (
                    x_out_per_rank[dst_rank] + kept[route_index]
                )
        # ranks fed by no kept route keep their input features, matching the
        # backbone's own fallback for ranks outside x_out_per_rank
        return x_out_per_rank

    @contextlib.contextmanager
    def coalition(self, mask: int):
        """Temporarily evaluate under the given coalition bitmask."""
        prev = self.coalition_mask
        self.coalition_mask = mask
        try:
            yield self
        finally:
            self.coalition_mask = prev

    def forward(self, batch):
        return self.backbone(batch)


def prune_backbone_(backbone: torch.nn.Module, mask: int) -> list[str]:
    """Permanently prune a TopoTune backbone to a coalition, in place.

    Removes dropped routes from ``neighborhoods``/``routes`` and deletes their
    per-layer GNN copies (freeing parameters). Kept routes keep their trained
    weights — the recipe continues training warm. Returns the kept
    neighborhood names in bit order.
    """
    kept_names = list(mask_to_coalition(mask))
    kept_indices = [
        i for i, name in enumerate(backbone.neighborhoods) if name in kept_names
    ]
    if not kept_indices:
        raise ValueError("refusing to prune to the empty coalition")

    backbone.neighborhoods = [
        backbone.neighborhoods[i] for i in kept_indices
    ]
    backbone.routes = [backbone.routes[i] for i in kept_indices]
    for layer_idx in range(backbone.layers):
        layer = backbone.graph_routes[layer_idx]
        backbone.graph_routes[layer_idx] = torch.nn.ModuleList(
            [layer[i] for i in kept_indices]
        )
    backbone.max_rank = max(max(route) for route in backbone.routes)
    return kept_names
