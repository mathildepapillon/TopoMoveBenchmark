"""Bit order, mask codecs, menu masks, and route-grammar consistency."""

import pytest

from toposhap.vocabulary import (
    ANCHOR,
    FULL_MASK,
    MENU,
    N_PLAYERS,
    NEIGHBORHOODS,
    coalition_to_mask,
    mask_size,
    mask_to_coalition,
    routes_from_neighborhoods,
)


def test_vocabulary_is_the_frozen_11():
    assert N_PLAYERS == 11
    assert FULL_MASK == 2047
    assert NEIGHBORHOODS[0] == "up_adjacency-0" == ANCHOR
    # the exact handoff order — changing it silently corrupts every frozen mask
    assert NEIGHBORHOODS == (
        "up_adjacency-0",
        "up_incidence-0",
        "down_incidence-1",
        "up_adjacency-1",
        "down_incidence-2",
        "2-up_adjacency-0",
        "down_adjacency-1",
        "up_incidence-1",
        "down_adjacency-2",
        "2-up_incidence-0",
        "2-down_incidence-2",
    )


def test_mask_roundtrip():
    for mask in [0, 1, 21, 26, 72, 129, 329, 479, 1024, FULL_MASK]:
        assert coalition_to_mask(mask_to_coalition(mask)) == mask


def test_menu_masks_decode():
    assert mask_to_coalition(MENU["HOPSE"].mask) == (
        "up_adjacency-0",
        "down_incidence-1",
        "down_incidence-2",
    )
    assert mask_to_coalition(MENU["CWN"].mask) == (
        "up_incidence-0",
        "up_adjacency-1",
        "down_incidence-2",
    )
    # every menu entry is a valid non-empty subset of the 11
    for entry in MENU.values():
        assert 0 < entry.mask <= FULL_MASK


def test_anchor_membership_across_menu():
    # the anchor (bit 0) is what anchored selection always keeps; menu entries
    # split on it — HOPSE/CCXN/SCN/SCCN contain it, CWN/CAN do not.
    has_anchor = {
        name: bool(entry.mask & 1) for name, entry in MENU.items()
    }
    assert has_anchor == {
        "HOPSE": True,
        "CWN": False,
        "CAN": False,
        "CCXN": True,
        "SCN": True,
        "SCCN": True,
    }


def test_routes_grammar():
    routes = routes_from_neighborhoods(list(NEIGHBORHOODS))
    assert routes == [
        [0, 0],  # up_adjacency-0
        [0, 1],  # up_incidence-0
        [1, 0],  # down_incidence-1
        [1, 1],  # up_adjacency-1
        [2, 1],  # down_incidence-2
        [0, 0],  # 2-up_adjacency-0
        [1, 1],  # down_adjacency-1
        [1, 2],  # up_incidence-1
        [2, 2],  # down_adjacency-2
        [0, 2],  # 2-up_incidence-0
        [2, 0],  # 2-down_incidence-2
    ]


def test_routes_match_topobench_resolver():
    topobench = pytest.importorskip("topobench")  # noqa: F841
    from topobench.utils.config_resolvers import (
        get_routes_from_neighborhoods,
    )

    ours = routes_from_neighborhoods(list(NEIGHBORHOODS))
    theirs = get_routes_from_neighborhoods(list(NEIGHBORHOODS))
    assert [list(r) for r in ours] == [list(r) for r in theirs]


def test_mask_size():
    assert mask_size(0) == 0
    assert mask_size(FULL_MASK) == 11
    assert mask_size(21) == 3
