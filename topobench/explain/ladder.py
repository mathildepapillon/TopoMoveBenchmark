"""Backward-elimination ladder: from a trained stem to a validated coalition.

This module orchestrates the neighborhood-selection pipeline as one fixed
protocol::

    all-players stem (5 epochs) -> backward-elimination ladder on the
    masked validation game -> train B rungs (rung 1 = binomial-se stop
    rung sqrt(p(1-p)/m); rungs 2..B = highest-cheap-value at unused
    sizes) -> validation selects.

The pipeline is family-agnostic. Players are whatever neighborhood-indexed
components the model family exposes — a GCCN's message-passing routes
(masked with :class:`topobench.explain.CoalitionMaskedBackbone`, pruned
with :func:`topobench.explain.prune_backbone_`) or a per-hop-encoding
model's encoding blocks (masked by zeroing their columns at model input).
:func:`run_ladder` never touches a trainer: the caller trains the
all-players stem first, then supplies two callables that close over it —
a cheap game scoring coalitions on the validation split, and a trainer
continuing the stem on a pruned coalition. Every rung choice is made from
the cheap game alone, before any rung is trained, so the choice of which
coalitions get a training budget can never peek at trained results.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from topobench.explain.games import CachedGame

__all__ = [
    "LadderResult",
    "LadderRung",
    "backward_elimination_ladder",
    "binomial_se",
    "run_ladder",
]


def binomial_se(p: float, m: int) -> float:
    """Standard error of an accuracy ``p`` measured on ``m`` samples.

    Used as the stop threshold of the ladder: a removal that costs less
    than one standard error of the full-coalition validation accuracy is
    indistinguishable from measurement noise.

    Parameters
    ----------
    p : float
        Accuracy-like game value in ``[0, 1]``.
    m : int
        Number of samples the value was measured on.

    Returns
    -------
    float
        ``sqrt(p * (1 - p) / m)``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"the binomial-se stop rule needs a game valued in [0, 1] "
            f"(an accuracy); got v(full) = {p}"
        )
    if m < 1:
        raise ValueError(f"val_size must be a positive sample count, got {m}")
    return math.sqrt(p * (1.0 - p) / m)


@dataclass
class LadderRung:
    """One rung of the elimination ladder.

    Parameters
    ----------
    mask : int
        Coalition bitmask of the rung.
    k : int
        Coalition size (number of set bits in ``mask``).
    cheap_value : float
        Value of the masked validation game at this coalition.
    val_score : float, optional
        Validation score after training this rung; ``None`` until (and
        unless) the rung is trained.
    """

    mask: int
    k: int
    cheap_value: float
    val_score: float | None = None


@dataclass
class LadderResult:
    """Everything :func:`run_ladder` decided and measured.

    Parameters
    ----------
    players : list of str
        Player names in bit order (bit ``i`` = ``players[i]``).
    ladder : list of LadderRung
        The full elimination path, one rung per size from ``n`` players
        down to 1.
    rungs : list of LadderRung
        The trained rungs in a-priori order (rung 1 first). These are the
        same objects as the corresponding ``ladder`` entries, with
        ``val_score`` filled in.
    selected : LadderRung
        The trained rung with the best validation score (ties toward
        rung 1).
    v_full : float
        Cheap game value of the grand coalition.
    threshold : float
        The binomial-se stop threshold ``sqrt(p(1-p)/m)``.
    n_game_evaluations : int
        Distinct cheap-game evaluations spent building the ladder.
    ties : list of dict
        Elimination ties, one dict per tied step with keys ``k_before``,
        ``bits``, and ``v`` (ties are broken toward the lowest bit).
    """

    players: list[str]
    ladder: list[LadderRung]
    rungs: list[LadderRung]
    selected: LadderRung
    v_full: float
    threshold: float
    n_game_evaluations: int
    ties: list[dict] = field(default_factory=list)

    @property
    def selected_players(self) -> list[str]:
        """Names of the players in the selected coalition.

        Returns
        -------
        list of str
            Player names in bit order.
        """
        return [
            name
            for i, name in enumerate(self.players)
            if self.selected.mask >> i & 1
        ]


def backward_elimination_ladder(
    game: Callable[[int], float],
    n_players: int,
    threshold: float,
) -> tuple[list[LadderRung], int, list[dict]]:
    """Build the full elimination ladder and locate the stop rung.

    Starting from the grand coalition, repeatedly remove the player whose
    removal hurts the cheap game least (ties toward the lowest bit) until
    one player remains. The stop rung is the first rung on the way down
    where the best removal would drop the value by more than
    ``threshold``; when that never happens, the floor ``k = 1`` rung is
    the stop rung.

    Parameters
    ----------
    game : Callable[[int], float]
        Cheap game over coalition bitmasks (the masked validation game).
    n_players : int
        Number of players.
    threshold : float
        Stop threshold on the rung-to-rung value drop.

    Returns
    -------
    tuple[list[LadderRung], int, list[dict]]
        The ladder (size ``n_players`` first, size 1 last), the index of
        the stop rung within it, and the recorded elimination ties.
    """
    mask = (1 << n_players) - 1
    base = float(game(mask))
    rungs = [LadderRung(mask=mask, k=n_players, cheap_value=base)]
    stop_index: int | None = None
    ties: list[dict] = []
    while bin(mask).count("1") > 1:
        best_bit, best_value = None, -math.inf
        tie_bits: list[int] = []
        for i in range(n_players):
            if not mask >> i & 1:
                continue
            value = float(game(mask & ~(1 << i)))
            if value > best_value:
                best_bit, best_value = i, value
                tie_bits = [i]
            elif value == best_value:
                tie_bits.append(i)
        if stop_index is None and base - best_value > threshold:
            stop_index = len(rungs) - 1  # stop rung found; keep building
        if len(tie_bits) > 1:
            ties.append(
                {
                    "k_before": bin(mask).count("1"),
                    "bits": tie_bits,
                    "v": float(best_value),
                }
            )
        mask &= ~(1 << best_bit)
        rungs.append(
            LadderRung(
                mask=mask, k=bin(mask).count("1"), cheap_value=best_value
            )
        )
        base = best_value
    if stop_index is None:
        stop_index = len(rungs) - 1  # never stopped: floor k = 1
    return rungs, stop_index, ties


def _choose_rungs(
    ladder: list[LadderRung], stop_index: int, budget_B: int
) -> list[LadderRung]:
    """Pick the a-priori rungs to train.

    Rung 1 is the stop rung. Rungs 2..B are, in order, the ladder rungs
    with the highest cheap value among sizes not already used (ties
    toward the smaller size).

    Parameters
    ----------
    ladder : list of LadderRung
        The full elimination ladder.
    stop_index : int
        Index of the stop rung within ``ladder``.
    budget_B : int
        Number of rungs to train. When it exceeds the number of distinct
        sizes on the ladder, every rung is trained.

    Returns
    -------
    list of LadderRung
        The chosen rungs, rung 1 first.
    """
    chosen = [ladder[stop_index]]
    used_sizes = {ladder[stop_index].k}
    while len(chosen) < budget_B:
        candidates = [r for r in ladder if r.k not in used_sizes]
        if not candidates:
            break
        rung = max(candidates, key=lambda r: (r.cheap_value, -r.k))
        chosen.append(rung)
        used_sizes.add(rung.k)
    return chosen


def run_ladder(
    train_fn: Callable[[int], float],
    game_fn: Callable[[int], float],
    players: Sequence[str],
    budget_B: int = 2,
    val_size: int | None = None,
) -> LadderResult:
    """Run the fixed selection pipeline on a trained all-players stem.

    Protocol (fixed, applied identically across datasets and model
    families)::

        all-players stem (5 epochs) -> backward-elimination ladder on the
        masked validation game -> train B rungs (rung 1 = binomial-se stop
        rung sqrt(p(1-p)/m); rungs 2..B = highest-cheap-value at unused
        sizes) -> validation selects.

    Concretely:

    1. The caller trains a stem containing every player (the reference
       setting is 5 epochs) — ``game_fn`` and ``train_fn`` close over it.
    2. The ladder is the backward-elimination path from the grand
       coalition down to ``k = 1`` on the cheap masked-validation game:
       at each step remove the player whose removal hurts the value
       least (ties toward the lowest bit).
    3. Rung 1 is the stop rung: the first rung on the way down where the
       best removal would drop the value by more than the binomial
       standard error ``sqrt(p(1-p)/m)`` with ``p = v(full)`` and ``m``
       the validation-set size (floor ``k = 1`` when elimination never
       triggers the stop). Rungs 2..B are the ladder rungs with the
       highest cheap value among sizes not already used (ties toward the
       smaller size).
    4. Every chosen rung is trained via ``train_fn`` (warm continuation
       from the same stem, in the reference setting), and the rung with
       the best returned validation score is selected (ties toward
       rung 1).

    All rung choices are fixed before any rung is trained.

    Parameters
    ----------
    train_fn : Callable[[int], float]
        Trains the coalition given by a bitmask (e.g. prune a copy of the
        stem and continue training) and returns its validation score.
    game_fn : Callable[[int], float]
        Cheap masked validation game: coalition bitmask -> value in
        ``[0, 1]`` (an accuracy). Memoized internally, so repeated masks
        cost one evaluation.
    players : Sequence[str]
        Player names in bit order — e.g. a GCCN's message-passing
        neighborhoods, or a per-hop-encoding model's encoding blocks.
    budget_B : int
        Number of rungs to train (the ``B`` in the protocol). Default 2.
    val_size : int
        Size of the validation split behind ``game_fn``, the ``m`` of the
        binomial-se stop rule.

    Returns
    -------
    LadderResult
        The full ladder, the trained rungs, and the selected coalition.
    """
    players = list(players)
    if not players:
        raise ValueError("run_ladder needs at least one player")
    if budget_B < 1:
        raise ValueError(f"budget_B must be at least 1, got {budget_B}")
    if val_size is None:
        raise ValueError(
            "val_size is required: the binomial-se stop rule is "
            "sqrt(p(1-p)/m) with m the validation-set size"
        )

    n_players = len(players)
    game = CachedGame(n_players=n_players, evaluate=game_fn)

    v_full = game((1 << n_players) - 1)
    threshold = binomial_se(v_full, val_size)
    ladder, stop_index, ties = backward_elimination_ladder(
        game, n_players, threshold
    )

    rungs = _choose_rungs(ladder, stop_index, budget_B)
    for rung in rungs:
        rung.val_score = float(train_fn(rung.mask))

    selected = rungs[0]
    for rung in rungs[1:]:
        if rung.val_score > selected.val_score:
            selected = rung

    return LadderResult(
        players=players,
        ladder=ladder,
        rungs=rungs,
        selected=selected,
        v_full=v_full,
        threshold=threshold,
        n_game_evaluations=game.calls,
        ties=ties,
    )
