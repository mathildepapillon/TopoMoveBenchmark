"""Coalition masking for HOPSE via encoding-block zeroing.

HOPSE's preprocessing transform (``hopse_ps_information``) computes one
positional/structural encoding per neighborhood route and CONCATENATES the
per-route blocks along the feature dimension per destination rank
(``aggregate_inter_nbhd`` uses ``torch.cat`` in route order). The cached data
carries ``x{rank}_{hop}`` where hop 0 is the copied initial features
(``copy_initial: True``) and hops 1..n_encodings are the per-encoding
concatenations. Dropping a neighborhood therefore means zeroing its column
slice in every ``x{rank}_{hop}`` (hop >= 1) whose rank is the route's
destination — #13's mechanism, validated there by an exhaustive 255-coalition
column-slicing check.

The masker monkey-wraps the model's ``HOPSEFeatureEncoder.forward`` (an
instance attribute, so checkpoints keep their state-dict keys) and zeroes the
dropped blocks on a CLONE of each input tensor before encoding.
"""

from __future__ import annotations

import contextlib

import torch

from topobench.data.utils import get_routes_from_neighborhoods

#: The 8-neighborhood HOPSE vocabulary in #13's player (bit) order
#: (analysis13.json per_dataset[*].players).
HOPSE_NEIGHBORHOODS: tuple[str, ...] = (
    "up_adjacency-0",  # bit 0
    "up_adjacency-1",  # bit 1
    "down_adjacency-1",  # bit 2
    "down_adjacency-2",  # bit 3
    "up_incidence-0",  # bit 4
    "up_incidence-1",  # bit 5
    "down_incidence-1",  # bit 6
    "down_incidence-2",  # bit 7
)
HOPSE_N_PLAYERS: int = len(HOPSE_NEIGHBORHOODS)
HOPSE_FULL_MASK: int = (1 << HOPSE_N_PLAYERS) - 1  # 255


class HopseCoalitionMasker:
    """Zero the encoding blocks of dropped neighborhoods at model input.

    Wraps a ``HOPSEFeatureEncoder``; while ``coalition_mask`` is not the full
    mask, every ``x{rank}_{hop}`` tensor (hop >= 1) has the column slices of
    dropped routes zeroed (on a clone) before encoding. Block layout is
    derived from the route list: routes are concatenated per destination rank
    in vocabulary order, each with equal width per encoding — asserted at
    runtime by divisibility against the encoder's ``in_channels``.
    """

    def __init__(self, feature_encoder, neighborhoods=HOPSE_NEIGHBORHOODS):
        self.encoder = feature_encoder
        self.neighborhoods = list(neighborhoods)
        routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.rank_routes: dict[int, list[int]] = {}
        for idx, (_, dst) in enumerate(routes):
            self.rank_routes.setdefault(dst, []).append(idx)
        self.coalition_mask: int = (1 << len(self.neighborhoods)) - 1
        self.full_mask: int = self.coalition_mask
        self._original_forward = feature_encoder.forward
        feature_encoder.forward = self._masked_forward

    def _masked_forward(self, data):
        if self.coalition_mask != self.full_mask:
            for rank, route_indices in self.rank_routes.items():
                if rank not in self.encoder.dimensions:
                    continue
                n_blocks = len(route_indices)
                dropped = [
                    position
                    for position, ridx in enumerate(route_indices)
                    if not self.coalition_mask >> ridx & 1
                ]
                if not dropped:
                    continue
                for hop in range(1, self.encoder.hops):
                    key = f"x{rank}_{hop}"
                    tensor = data[key]
                    if tensor.shape[1] % n_blocks != 0:
                        raise AssertionError(
                            f"{key}: width {tensor.shape[1]} not divisible "
                            f"by {n_blocks} routes — block layout assumption "
                            "violated"
                        )
                    width = tensor.shape[1] // n_blocks
                    tensor = tensor.clone()
                    for position in dropped:
                        tensor[:, position * width:(position + 1) * width] = 0
                    data[key] = tensor
        return self._original_forward(data)

    @contextlib.contextmanager
    def coalition(self, mask: int):
        """Temporarily evaluate under the given coalition bitmask."""
        prev = self.coalition_mask
        self.coalition_mask = mask
        try:
            yield self
        finally:
            self.coalition_mask = prev

    def fix(self, mask: int) -> None:
        """Permanently fix the coalition (the HOPSE analogue of pruning)."""
        self.coalition_mask = mask

    def restore(self) -> None:
        """Unhook: give the encoder its original forward back."""
        self.encoder.forward = self._original_forward


@torch.no_grad()
def block_widths(feature_encoder, neighborhoods=HOPSE_NEIGHBORHOODS):
    """Per-(rank, hop) block widths implied by encoder.in_channels (debug)."""
    routes = get_routes_from_neighborhoods(list(neighborhoods))
    rank_counts: dict[int, int] = {}
    for _, dst in routes:
        rank_counts[dst] = rank_counts.get(dst, 0) + 1
    out = {}
    for rank in feature_encoder.dimensions:
        n = rank_counts.get(rank, 0)
        for hop in range(1, feature_encoder.hops):
            total = feature_encoder.in_channels[rank][hop]
            out[(rank, hop)] = (total, n, total // n if n else None)
    return out
