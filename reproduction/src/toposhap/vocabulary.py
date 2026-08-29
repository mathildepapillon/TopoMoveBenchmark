"""The 11-neighborhood vocabulary and its bitmask encoding.

Bit order IS player order everywhere in this codebase, in every result file,
and in the paper. Never reorder this list: subset bitmasks in frozen results
(``analysis12``, ``pooled``, ``menu12``, ...) are integers over exactly this
ordering. ``up_adjacency-0`` (atom<->atom on molecule graphs) is bit 0 and is
"the anchor" in anchored selection.

Neighborhood strings follow TopoBench's grammar
(``topobench.utils.config_resolvers.get_routes_from_neighborhoods``):
``[r-]{up|down}_{adjacency|incidence}-{src_rank}``, where the optional leading
``r-`` is the hop/rank offset (default 1).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Player order. Bit i of a subset mask corresponds to NEIGHBORHOODS[i].
NEIGHBORHOODS: tuple[str, ...] = (
    "up_adjacency-0",  # bit 0 — the anchor (atom<->atom)
    "up_incidence-0",  # bit 1
    "down_incidence-1",  # bit 2
    "up_adjacency-1",  # bit 3
    "down_incidence-2",  # bit 4
    "2-up_adjacency-0",  # bit 5
    "down_adjacency-1",  # bit 6
    "up_incidence-1",  # bit 7
    "down_adjacency-2",  # bit 8
    "2-up_incidence-0",  # bit 9
    "2-down_incidence-2",  # bit 10
)

N_PLAYERS: int = len(NEIGHBORHOODS)
FULL_MASK: int = (1 << N_PLAYERS) - 1  # 2047
ANCHOR_BIT: int = 0
ANCHOR: str = NEIGHBORHOODS[ANCHOR_BIT]

_INDEX: dict[str, int] = {name: i for i, name in enumerate(NEIGHBORHOODS)}


def routes_from_neighborhoods(neighborhoods: list[str]) -> list[list[int]]:
    """(src_rank, dst_rank) routes, mirroring TopoBench's resolver exactly."""
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1
        if "incidence" in neighborhood:
            route = (
                [src_rank, src_rank - r]
                if "down" in neighborhood
                else [src_rank, src_rank + r]
            )
        elif "adjacency" in neighborhood:
            route = [src_rank, src_rank]
        else:
            raise ValueError(f"Invalid neighborhood {neighborhood}")
        routes.append(route)
    return routes


def coalition_to_mask(names: list[str] | tuple[str, ...]) -> int:
    """Encode a set of neighborhood names as an integer bitmask."""
    mask = 0
    for name in names:
        mask |= 1 << _INDEX[name]
    return mask


def mask_to_coalition(mask: int) -> tuple[str, ...]:
    """Decode an integer bitmask into neighborhood names, bit order."""
    if not 0 <= mask <= FULL_MASK:
        raise ValueError(f"mask {mask} out of range for {N_PLAYERS} players")
    return tuple(NEIGHBORHOODS[i] for i in range(N_PLAYERS) if mask >> i & 1)


def mask_size(mask: int) -> int:
    """Number of neighborhoods in the coalition."""
    return bin(mask).count("1")


@dataclass(frozen=True)
class MenuEntry:
    """A fixed neighborhood set from the TopoBench architecture menu."""

    mask: int
    provenance: str  # which published architecture this set corresponds to


#: The 6-entry TopoBench menu (fixed sets with provenance, cf. menu12.json).
MENU: dict[str, MenuEntry] = {
    "HOPSE": MenuEntry(mask=21, provenance="HOPSE default"),
    "CWN": MenuEntry(mask=26, provenance="CWN"),
    "CAN": MenuEntry(mask=72, provenance="CAN/CCCN/SAN"),
    "CCXN": MenuEntry(mask=129, provenance="CCXN"),
    "SCN": MenuEntry(mask=329, provenance="SCN"),
    "SCCN": MenuEntry(mask=479, provenance="SCCN/SCCNN"),
}
