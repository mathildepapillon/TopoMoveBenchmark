"""The backward-elimination ladder on synthetic games with known optima."""

import math

import pytest

from topobench.explain import (
    backward_elimination_ladder,
    binomial_se,
    run_ladder,
)

# Additive toy game: v(mask) = 0.5 + sum of per-player bonuses. Players
# "a" and "d" carry the task (+0.2 each); "b", "c", "e" are almost-null
# (+0.001..0.003, below any reasonable noise floor); "f" actively hurts
# (-0.01). The unique smallest coalition preserving the value is {a, d}.
PLAYERS = ["a", "b", "c", "d", "e", "f"]
BONUS = [0.2, 0.001, 0.002, 0.2, 0.003, -0.01]
VAL_SIZE = 100


def toy_game(mask):
    """Value of a coalition in the additive toy game.

    Parameters
    ----------
    mask : int
        Coalition bitmask.

    Returns
    -------
    float
        Base value 0.5 plus the bonuses of the present players.
    """
    return 0.5 + sum(b for i, b in enumerate(BONUS) if mask >> i & 1)


V_FULL = toy_game((1 << len(PLAYERS)) - 1)  # 0.896
OPTIMUM = 0b001001  # {a, d}


def toy_train(mask):
    """Deterministic stand-in for rung training: cheap value minus 4%.

    Parameters
    ----------
    mask : int
        Coalition bitmask.

    Returns
    -------
    float
        A validation score ordered like the cheap game.
    """
    return toy_game(mask) - 0.04


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------


def test_ladder_walks_every_size_and_removes_null_players_first():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    assert [r.k for r in result.ladder] == [6, 5, 4, 3, 2, 1]
    # the harmful player goes first, then the almost-null ones by size
    assert [r.mask for r in result.ladder] == [
        0b111111,
        0b011111,
        0b011101,
        0b011001,
        0b001001,
        0b001000,
    ]


def test_stop_rule_triggers_at_the_true_optimum():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    rung1 = result.rungs[0]
    # removals of the almost-null players cost <= 0.003 < threshold; the
    # first removal that would touch {a, d} costs 0.2 > threshold
    assert result.threshold == pytest.approx(
        math.sqrt(V_FULL * (1 - V_FULL) / VAL_SIZE)
    )
    assert 0.003 < result.threshold < 0.2
    assert rung1.mask == OPTIMUM
    assert rung1.k == 2
    assert rung1.cheap_value == pytest.approx(0.9)


def test_rung2_is_highest_cheap_value_at_an_unused_size():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    rung2 = result.rungs[1]
    # dropping the harmful player gives the ladder's best cheap value
    assert rung2.mask == 0b011111
    assert rung2.k == 5
    assert rung2.cheap_value == pytest.approx(V_FULL + 0.01)


def test_validation_selects_the_optimum():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    # trained scores are ordered like the cheap game: rung2 (k=5) wins on
    # raw score 0.906 - 0.04 > 0.9 - 0.04, so selection must follow the
    # trained validation scores, not the rung order
    assert result.selected is result.rungs[1]

    # with training noise that punishes the bloated coalition, the stop
    # rung wins and the ladder lands exactly on the true optimum
    def train_small_wins(mask):
        return toy_game(mask) - 0.01 * bin(mask).count("1")

    result = run_ladder(
        train_small_wins, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    assert result.selected.mask == OPTIMUM
    assert result.selected_players == ["a", "d"]


def test_validation_tie_goes_to_rung1():
    result = run_ladder(
        lambda mask: 0.8, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    assert result.selected is result.rungs[0]
    assert result.selected.mask == OPTIMUM


def test_elimination_tie_breaks_toward_the_lowest_bit_and_is_recorded():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    # at k=2 removing "a" and removing "d" tie at 0.7; "a" (bit 0) goes
    assert result.ladder[-1].mask == 0b001000
    assert result.ties == [
        {"k_before": 2, "bits": [0, 3], "v": pytest.approx(0.7)}
    ]


def test_game_evaluations_are_cached_and_counted():
    calls = []

    def counting_game(mask):
        calls.append(mask)
        return toy_game(mask)

    n = len(PLAYERS)
    result = run_ladder(
        toy_train, counting_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    # 1 grand coalition + k evaluations per elimination step, no repeats
    assert result.n_game_evaluations == n * (n + 1) // 2
    assert len(calls) == len(set(calls)) == result.n_game_evaluations


def test_rung_choices_are_a_priori():
    trained = []

    def recording_train(mask):
        trained.append(mask)
        return 0.5

    result = run_ladder(
        recording_train, toy_game, PLAYERS, budget_B=2, val_size=VAL_SIZE
    )
    # only the two chosen rungs are trained, rung 1 first, and the choice
    # was already fixed by the cheap game
    assert trained == [OPTIMUM, 0b011111]
    assert [r.val_score for r in result.rungs] == [0.5, 0.5]


def test_budget_beyond_two_fills_unused_sizes_in_cheap_value_order():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=4, val_size=VAL_SIZE
    )
    assert [r.k for r in result.rungs] == [2, 5, 4, 3]
    cheap = [r.cheap_value for r in result.rungs[1:]]
    assert cheap == sorted(cheap, reverse=True)


def test_budget_exceeding_ladder_trains_every_size_once():
    result = run_ladder(
        toy_train, toy_game, PLAYERS, budget_B=100, val_size=VAL_SIZE
    )
    assert sorted(r.k for r in result.rungs) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Stop-rule edge cases
# ---------------------------------------------------------------------------


def test_flat_game_floors_at_k_equals_one():
    result = run_ladder(
        lambda mask: 0.75,
        lambda mask: 0.7,
        ["x", "y", "z"],
        budget_B=2,
        val_size=50,
    )
    # no removal ever costs anything: elimination never stops, floor k=1
    assert result.rungs[0].k == 1
    # rung 2 ties on cheap value across sizes; the smaller size wins
    assert result.rungs[1].k == 2


def test_stop_rung_is_the_first_trigger_on_the_way_down():
    ladder, stop_index, _ = backward_elimination_ladder(
        toy_game, len(PLAYERS), threshold=0.0025
    )
    # with a tighter threshold the +0.003 player already triggers the
    # stop at k=3, even though later drops are larger
    assert ladder[stop_index].k == 3
    assert ladder[stop_index].mask == 0b011001


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_games_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"valued in \[0, 1\]"):
        run_ladder(
            toy_train, lambda mask: 1.5, PLAYERS, budget_B=2, val_size=100
        )


def test_rejects_bad_budget_players_and_val_size():
    with pytest.raises(ValueError, match="budget_B"):
        run_ladder(toy_train, toy_game, PLAYERS, budget_B=0, val_size=100)
    with pytest.raises(ValueError, match="at least one player"):
        run_ladder(toy_train, toy_game, [], budget_B=2, val_size=100)
    with pytest.raises(ValueError, match="val_size is required"):
        run_ladder(toy_train, toy_game, PLAYERS, budget_B=2)
    with pytest.raises(ValueError, match="positive sample count"):
        run_ladder(toy_train, toy_game, PLAYERS, budget_B=2, val_size=0)


def test_binomial_se_matches_the_formula():
    assert binomial_se(0.5, 25) == pytest.approx(0.1)
    assert binomial_se(0.0, 10) == 0.0
    assert binomial_se(1.0, 10) == 0.0
