"""Coalition masking of neighborhoods in TopoTune (GCCN) backbones.

A neighborhood game needs the model's output under an arbitrary coalition
of message-passing neighborhoods. The masking point is
``aggregate_inter_nbhd``: per layer, each route's GNN output is summed into
its destination rank, so dropping a neighborhood means dropping its route's
contribution from that sum — exactly what "the player is absent" means for
a message-passing route.

Player (bit) order is the order of ``backbone.neighborhoods`` as
configured, whatever its length: bit ``i`` of a coalition mask corresponds
to ``backbone.neighborhoods[i]``.

Two uses:

* :class:`CoalitionMaskedBackbone` — inference-time wrapper for
  attribution games on a trained backbone.
* :func:`prune_backbone_` — permanent in-place prune to a coalition, so
  continued training pays only for kept neighborhoods while keeping their
  trained weights (warm start).
"""

import contextlib
from collections.abc import Sequence

import torch


def coalition_to_mask(
    names: Sequence[str], neighborhoods: Sequence[str]
) -> int:
    """Encode a set of neighborhood names as an integer bitmask.

    Parameters
    ----------
    names : Sequence[str]
        Neighborhood names to include in the coalition.
    neighborhoods : Sequence[str]
        Player order: bit ``i`` corresponds to ``neighborhoods[i]``.

    Returns
    -------
    int
        Coalition bitmask over ``neighborhoods``.
    """
    index = {name: i for i, name in enumerate(neighborhoods)}
    mask = 0
    for name in names:
        if name not in index:
            raise ValueError(
                f"unknown neighborhood {name!r}; expected one of "
                f"{list(neighborhoods)}"
            )
        mask |= 1 << index[name]
    return mask


def mask_to_coalition(
    mask: int, neighborhoods: Sequence[str]
) -> tuple[str, ...]:
    """Decode an integer bitmask into neighborhood names, in bit order.

    Parameters
    ----------
    mask : int
        Coalition bitmask over ``neighborhoods``.
    neighborhoods : Sequence[str]
        Player order: bit ``i`` corresponds to ``neighborhoods[i]``.

    Returns
    -------
    tuple[str, ...]
        Names of the neighborhoods present in the coalition.
    """
    n = len(neighborhoods)
    if not 0 <= mask < (1 << n):
        raise ValueError(f"mask {mask} out of range for {n} players")
    return tuple(neighborhoods[i] for i in range(n) if mask >> i & 1)


class CoalitionMaskedBackbone(torch.nn.Module):
    """Wrap a TopoTune backbone so forward() respects a coalition mask.

    The wrapper replaces the backbone's ``aggregate_inter_nbhd`` with a
    version that drops the contribution of routes whose neighborhood bit is
    absent from :attr:`coalition_mask`. The backbone's neighborhood list is
    captured at wrap time and exposed as :attr:`players`; bit ``i`` of a
    coalition mask corresponds to ``players[i]``. The default mask is the
    grand coalition, under which the wrapper is equivalent to the unwrapped
    backbone.

    Parameters
    ----------
    backbone : torch.nn.Module
        A TopoTune backbone exposing ``neighborhoods``, ``routes`` and
        ``aggregate_inter_nbhd``.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        #: Player order captured at wrap time; bit i = players[i].
        self.players: list[str] = list(backbone.neighborhoods)
        self.coalition_mask: int = self.full_mask
        self._original_aggregate = backbone.aggregate_inter_nbhd
        backbone.aggregate_inter_nbhd = self._masked_aggregate

    @property
    def full_mask(self) -> int:
        """Return the grand-coalition bitmask over the wrapped players.

        Returns
        -------
        int
            Bitmask with one bit set per player.
        """
        return (1 << len(self.players)) - 1

    def _masked_aggregate(self, x_out_per_route: dict) -> dict:
        """Aggregate per-route outputs, dropping masked-out routes.

        Parameters
        ----------
        x_out_per_route : dict
            Per-route GNN outputs, keyed by route index.

        Returns
        -------
        dict
            Per-rank aggregated outputs over the active coalition. Ranks
            fed by no kept route are absent, matching the backbone's own
            fallback of keeping their input features.
        """
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
        return x_out_per_rank

    @contextlib.contextmanager
    def coalition(self, mask: int):
        """Temporarily evaluate under the given coalition bitmask.

        Parameters
        ----------
        mask : int
            Coalition bitmask over :attr:`players`.

        Yields
        ------
        CoalitionMaskedBackbone
            This wrapper, with :attr:`coalition_mask` set to ``mask``.
        """
        if not 0 <= mask <= self.full_mask:
            raise ValueError(
                f"mask {mask} out of range for {len(self.players)} players"
            )
        prev = self.coalition_mask
        self.coalition_mask = mask
        try:
            yield self
        finally:
            self.coalition_mask = prev

    def forward(self, batch):
        """Run the wrapped backbone under the active coalition.

        Parameters
        ----------
        batch : Complex or ComplexBatch(Complex)
            The input data.

        Returns
        -------
        dict
            The output hidden states of the backbone per rank.
        """
        return self.backbone(batch)


def prune_backbone_(backbone: torch.nn.Module, mask: int) -> list[str]:
    """Permanently prune a TopoTune backbone to a coalition, in place.

    Removes dropped routes from ``neighborhoods``/``routes`` and deletes
    their per-layer GNN copies (freeing parameters). Kept routes keep their
    trained weights, so continued training warm-starts from the pruned
    model. Bit ``i`` of ``mask`` corresponds to ``backbone.neighborhoods[i]``
    at call time.

    Parameters
    ----------
    backbone : torch.nn.Module
        A TopoTune backbone exposing ``neighborhoods``, ``routes``,
        ``layers`` and ``graph_routes``.
    mask : int
        Coalition bitmask of the neighborhoods to keep.

    Returns
    -------
    list[str]
        The kept neighborhood names, in bit order.
    """
    n = len(backbone.neighborhoods)
    if not 0 <= mask < (1 << n):
        raise ValueError(f"mask {mask} out of range for {n} players")
    kept_indices = [i for i in range(n) if mask >> i & 1]
    if not kept_indices:
        raise ValueError("refusing to prune to the empty coalition")

    backbone.neighborhoods = [backbone.neighborhoods[i] for i in kept_indices]
    backbone.routes = [backbone.routes[i] for i in kept_indices]
    for layer_idx in range(backbone.layers):
        layer = backbone.graph_routes[layer_idx]
        backbone.graph_routes[layer_idx] = torch.nn.ModuleList(
            [layer[i] for i in kept_indices]
        )
    backbone.max_rank = max(max(route) for route in backbone.routes)
    return list(backbone.neighborhoods)
