"""Subset selection: from attributions (or the game itself) to a coalition.

Fixed-size selectors, in ascending order of robustness to redundancy:

* :func:`top_k_mask` — plain top-k by Shapley value. Known to be the wrong
  rule under redundancy (Fryer, Strumke & Nguyen 2021): perfect substitutes
  split their shared payoff and can both be selected, leaving real marginal
  value on the table.
* :func:`anchored_top_k_mask` — always keep a designated anchor player,
  fill the remaining slots by Shapley value.
* :func:`greedy_mask` — greedy forward selection on the game itself.
  Handles redundancy structurally (after one substitute is added the
  other's marginal collapses) and carries the Nemhauser, Wolsey & Fisher
  (1978) ``(1 - 1/e)`` guarantee iff the game is submodular — which
  :func:`submodularity_census` measures rather than assumes.

Automatic-size rules (:func:`threshold_at_zero`, :func:`cost_regularized`,
:func:`greedy_with_noise_stop`, :func:`smallest_sufficient_coalition`)
return a coalition whose size falls out of the rule rather than being an
input.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from topobench.explain.games import TabulatedGame


def top_k_mask(phi: np.ndarray, k: int) -> int:
    """Select the k players with the largest Shapley values.

    Ties break toward the lower bit index, so the pick is deterministic
    across runs.

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.
    k : int
        Number of players to select.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    order = np.lexsort((np.arange(len(phi)), -phi))
    mask = 0
    for i in order[:k]:
        mask |= 1 << int(i)
    return mask


def anchored_top_k_mask(phi: np.ndarray, k: int, anchor_bit: int) -> int:
    """Select top-k players while always including an anchor player.

    The anchor occupies one of the k slots; the rest fill by Shapley
    value. Useful when domain knowledge says one player must stay (e.g. a
    neighborhood the task is known to depend on).

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.
    k : int
        Number of players to select (including the anchor).
    anchor_bit : int
        Bit index of the player that must be kept.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    if k < 1:
        raise ValueError("anchored selection needs k >= 1")
    if not 0 <= anchor_bit < len(phi):
        raise ValueError(
            f"anchor_bit {anchor_bit} out of range for {len(phi)} players"
        )
    mask = 1 << anchor_bit
    order = np.lexsort((np.arange(len(phi)), -phi))
    for i in order:
        if bin(mask).count("1") >= k:
            break
        if int(i) != anchor_bit:
            mask |= 1 << int(i)
    return mask


def greedy_mask(
    game: Callable[[int], float],
    n_players: int,
    k: int,
    start_mask: int = 0,
    return_trace: bool = False,
):
    """Select up to k players by greedy forward selection on the game.

    At each step, adds the player with the largest marginal gain in game
    value. ``start_mask`` seeds the coalition (pass ``1 << anchor_bit`` for
    an anchored greedy).

    Parameters
    ----------
    game : Callable[[int], float]
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    k : int
        Target coalition size (including any seeded players).
    start_mask : int
        Initial coalition to grow from.
    return_trace : bool
        Whether to also return the per-step ``(bit, gain)`` list, which
        automatic-size rules consume.

    Returns
    -------
    int or tuple[int, list[tuple[int, float]]]
        The selected coalition bitmask; with ``return_trace=True``, also
        the greedy trace.
    """
    mask = start_mask
    trace: list[tuple[int, float]] = []
    while bin(mask).count("1") < k:
        best_bit, best_gain = None, -np.inf
        base = game(mask)
        for i in range(n_players):
            bit = 1 << i
            if mask & bit:
                continue
            gain = game(mask | bit) - base
            if gain > best_gain:
                best_bit, best_gain = i, gain
        if best_bit is None:
            break
        mask |= 1 << best_bit
        trace.append((best_bit, float(best_gain)))
    if return_trace:
        return mask, trace
    return mask


def threshold_at_zero(phi: np.ndarray) -> int:
    """Select every player with a strictly positive Shapley value.

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    mask = 0
    for i, p in enumerate(phi):
        if p > 0:
            mask |= 1 << i
    return mask


def cost_regularized(phi: np.ndarray, cost_per_player: float) -> int:
    """Select players whose Shapley value exceeds a per-player cost.

    Equivalent to maximising ``sum(phi_i) - c * k`` over top-k prefixes
    when ``phi`` is sorted, but stated per player so ties and negative
    values behave sensibly.

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.
    cost_per_player : float
        Cost charged for each kept player.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    mask = 0
    for i, p in enumerate(phi):
        if p > cost_per_player:
            mask |= 1 << i
    return mask


def greedy_with_noise_stop(
    game: Callable[[int], float],
    n_players: int,
    noise_sd: float,
    start_mask: int = 0,
    max_k: int | None = None,
) -> int:
    """Run greedy selection, stopping below the evaluation-noise floor.

    ``noise_sd`` should be the measured noise of the game value (e.g. the
    standard deviation across retrainings of a fixed coalition); the rule
    stops once marginal gains are indistinguishable from that noise.

    Parameters
    ----------
    game : Callable[[int], float]
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    noise_sd : float
        Noise floor below which a marginal gain is not credited.
    start_mask : int
        Initial coalition to grow from.
    max_k : int, optional
        Cap on the coalition size; defaults to ``n_players``.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    limit = max_k if max_k is not None else n_players
    mask, trace = greedy_mask(
        game, n_players, k=limit, start_mask=start_mask, return_trace=True
    )
    kept = start_mask
    for bit, gain in trace:
        if gain < noise_sd:
            break
        kept |= 1 << bit
    return kept


def smallest_sufficient_coalition(
    game: Callable[[int], float],
    n_players: int,
    tolerance: float,
    phi: np.ndarray | None = None,
) -> int:
    """Find the smallest prefix within tolerance of the grand coalition.

    Candidate prefixes are ordered by Shapley value when ``phi`` is given
    (cheap: n game evaluations), otherwise by greedy trace. Falls back to
    the grand coalition if nothing smaller suffices.

    Parameters
    ----------
    game : Callable[[int], float]
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    tolerance : float
        Maximum allowed drop below the grand coalition's value.
    phi : np.ndarray, optional
        Attribution per player used to order candidate prefixes.

    Returns
    -------
    int
        Coalition bitmask of the selected players.
    """
    full = (1 << n_players) - 1
    target = game(full) - tolerance
    if phi is not None:
        for k in range(1, n_players + 1):
            mask = top_k_mask(phi, k)
            if game(mask) >= target:
                return mask
        return full
    _, trace = greedy_mask(game, n_players, k=n_players, return_trace=True)
    mask = 0
    for bit, _ in trace:
        mask |= 1 << bit
        if game(mask) >= target:
            return mask
    return full


@dataclass
class SubmodularityCensus:
    """Census of submodularity violations over a tabulated game.

    Parameters
    ----------
    triples_checked : int
        Number of ``(S, T, i)`` triples checked.
    violations : int
        Number of triples violating submodularity.
    violation_rate : float
        Fraction of checked triples that violate submodularity.
    worst_violation : float
        Most negative value of ``(v(S+i) - v(S)) - (v(T+i) - v(T))``.
    mean_violation : float
        Mean magnitude among violations (0 if none).
    """

    triples_checked: int
    violations: int
    violation_rate: float
    worst_violation: float
    mean_violation: float


def submodularity_census(
    game: TabulatedGame,
    max_triples: int = 2_000_000,
    seed: int = 0,
    atol: float = 1e-12,
) -> SubmodularityCensus:
    """Measure submodularity violations of an exhaustively tabulated game.

    The greedy selector's ``(1 - 1/e)`` guarantee holds iff the game is
    submodular: for every ``(S, T, i)`` with ``S`` a subset of ``T`` and
    ``i`` outside ``T``, ``v(S+i) - v(S) >= v(T+i) - v(T)``. Enumerating
    all triples is infeasible beyond small ``n``, so the census checks
    non-empty triples only (``S`` non-empty), exhaustively for small
    tables or by uniform sampling for large ones, and reports the
    violation rate and magnitude distribution.

    Parameters
    ----------
    game : TabulatedGame
        Exhaustively tabulated game to audit.
    max_triples : int
        Above this many candidate triples, sample uniformly instead of
        enumerating (reported counts are then estimates).
    seed : int
        Seed for the sampling regime.
    atol : float
        Tolerance separating violations from float noise; for measured
        (e.g. retrained) landscapes pass the evaluation noise instead.

    Returns
    -------
    SubmodularityCensus
        Violation counts, rate, and magnitude statistics.
    """
    n = game.n_players
    masks = sorted(game.values)
    rng = np.random.default_rng(seed)

    def random_triple():
        while True:
            t_mask = int(rng.integers(1, 1 << n))
            free = [i for i in range(n) if not t_mask >> i & 1]
            if not free:
                continue
            i = int(rng.choice(free))
            # random non-empty strict subset of t_mask
            bits = [b for b in range(n) if t_mask >> b & 1]
            if len(bits) < 2:
                continue
            keep = rng.random(len(bits)) < 0.5
            s_mask = sum(1 << b for b, k in zip(bits, keep, strict=True) if k)
            if s_mask == 0 or s_mask == t_mask:
                continue
            return s_mask, t_mask, i

    def exhaustive_triples():
        for t_mask in masks:
            if t_mask == 0:
                continue
            free = [i for i in range(n) if not t_mask >> i & 1]
            sub = t_mask
            while True:
                sub = (sub - 1) & t_mask
                if sub == 0:
                    break
                for i in free:
                    yield sub, t_mask, i

    checked = violations = 0
    worst = 0.0
    viol_sum = 0.0

    if not game.is_exhaustive:
        raise ValueError("submodularity census needs an exhaustive game table")

    total_estimate = sum(
        (2 ** bin(t).count("1") - 2) * (n - bin(t).count("1")) for t in masks
    )
    use_sampling = total_estimate > max_triples

    source = (
        (random_triple() for _ in range(max_triples))
        if use_sampling
        else exhaustive_triples()
    )
    v = game.values
    for s_mask, t_mask, i in source:
        bit = 1 << i
        gap = (v[s_mask | bit] - v[s_mask]) - (v[t_mask | bit] - v[t_mask])
        checked += 1
        if gap < -atol:
            violations += 1
            viol_sum += -gap
            worst = min(worst, gap)

    return SubmodularityCensus(
        triples_checked=checked,
        violations=violations,
        violation_rate=violations / checked if checked else 0.0,
        worst_violation=worst,
        mean_violation=viol_sum / violations if violations else 0.0,
    )
