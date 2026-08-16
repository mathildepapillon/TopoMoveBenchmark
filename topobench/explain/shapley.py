"""Exact and sampled Shapley values for cooperative games.

Exact computation enumerates all 2^n coalitions, so it is exact by
construction and tractable for small player sets (roughly n <= 20). The
permutation-sampled estimator scales to larger player sets: one pass is one
uniformly random permutation of the players, walking it front to back yields
one marginal contribution per player, and a budget of P passes costs at most
``n * P + 1`` distinct game evaluations (far fewer in practice with caching,
since permutation prefixes repeat).

The pairwise Shapley interaction index follows Grabisch & Roubens (1999),
"An axiomatic approach to the concept of interaction among players in
cooperative games".
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

from topobench.explain.games import Game


def _shapley_weights(n: int) -> np.ndarray:
    """Compute the Shapley weight of a marginal from each coalition size.

    Parameters
    ----------
    n : int
        Number of players.

    Returns
    -------
    np.ndarray
        Array ``w`` with ``w[s] = s! (n-s-1)! / n!`` for ``s`` in
        ``[0, n)``.
    """
    return np.array(
        [
            math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            for s in range(n)
        ]
    )


def shapley_values(game: Game, n_players: int) -> np.ndarray:
    """Compute exact Shapley values by enumerating all coalitions.

    Implements ``phi_i = sum_S w(|S|) [v(S + i) - v(S)]`` over all
    coalitions ``S`` not containing ``i``, evaluating the game on each of
    the 2^n masks exactly once.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.

    Returns
    -------
    np.ndarray
        Shapley value per player, shape ``[n_players]``.
    """
    n = n_players
    values = np.array([game(mask) for mask in range(1 << n)])
    weights = _shapley_weights(n)
    sizes = np.array([bin(m).count("1") for m in range(1 << n)])

    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        without = np.array([m for m in range(1 << n) if not m & bit])
        phi[i] = np.sum(
            weights[sizes[without]] * (values[without | bit] - values[without])
        )
    return phi


def shapley_interaction(
    game: Game, n_players: int, order: int = 2
) -> dict[frozenset[int], float]:
    """Compute the exact Shapley interaction index of Grabisch & Roubens.

    For every subset ``T`` of the given order, computes
    ``I(T) = sum_{S subset of N minus T} w_{|T|}(|S|) * delta_T v(S)``,
    where ``delta_T v(S)`` is the discrete derivative
    ``sum_{L subset of T} (-1)^{|T| - |L|} v(S union L)`` and
    ``w_t(s) = s! (n - s - t)! / (n - t + 1)!``.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    order : int
        Interaction order (size of the subsets ``T``).

    Returns
    -------
    dict[frozenset[int], float]
        Mapping from each order-``order`` player subset to its interaction
        index.
    """
    n = n_players
    t = order
    values = {mask: game(mask) for mask in range(1 << n)}

    def w(s: int) -> float:
        return (
            math.factorial(s)
            * math.factorial(n - s - t)
            / math.factorial(n - t + 1)
        )

    out: dict[frozenset[int], float] = {}
    for T in itertools.combinations(range(n), t):
        t_mask = 0
        for i in T:
            t_mask |= 1 << i
        rest = [m for m in range(1 << n) if not m & t_mask]
        total = 0.0
        for s_mask in rest:
            s = bin(s_mask).count("1")
            delta = 0.0
            for r in range(t + 1):
                for L in itertools.combinations(T, r):
                    l_mask = 0
                    for i in L:
                        l_mask |= 1 << i
                    delta += (-1) ** (t - r) * values[s_mask | l_mask]
            total += w(s) * delta
        out[frozenset(T)] = total
    return out


def efficiency_gap(phi: np.ndarray, game: Game, n_players: int) -> float:
    """Compute the efficiency-axiom residual of an attribution.

    Returns ``|sum_i phi_i - (v(N) - v(empty))|``, which is zero (up to
    float precision) for exact Shapley values. Estimators report it as a
    convergence diagnostic; :func:`topobench.explain.axioms.check_efficiency`
    turns it into an assertion.

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.

    Returns
    -------
    float
        Absolute efficiency gap.
    """
    full = (1 << n_players) - 1
    return float(abs(phi.sum() - (game(full) - game(0))))


@dataclass
class SampledAttribution:
    """Result of a permutation-sampled Shapley estimation run.

    Parameters
    ----------
    phi : np.ndarray
        Point estimate of the Shapley value per player.
    stderr : np.ndarray
        Standard error of the mean per player, across passes.
    passes : int
        Number of permutations consumed.
    evaluations : int
        Number of distinct game evaluations (the honest cost).
    """

    phi: np.ndarray
    stderr: np.ndarray
    passes: int
    evaluations: int


def sampled_shapley(
    game: Game,
    n_players: int,
    passes: int = 128,
    seed: int | None = None,
    antithetic: bool = True,
) -> SampledAttribution:
    """Estimate Shapley values by permutation sampling.

    Each pass draws a uniformly random permutation of the players and
    records one marginal contribution per player. With ``antithetic=True``
    each drawn permutation is paired with its reverse, which cancels a
    first-order term of the variance at no extra evaluation cost beyond the
    shared cache.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    passes : int
        Number of permutations to average over.
    seed : int, optional
        Seed for the random number generator.
    antithetic : bool
        Whether to pair each permutation with its reverse. Note that exact
        substitutes receive identical estimates under antithetic pairing,
        so synthetic stability probes should pass ``antithetic=False``.

    Returns
    -------
    SampledAttribution
        Point estimates, per-player standard errors, and cost accounting.
    """
    rng = np.random.default_rng(seed)
    n = n_players

    cache: dict[int, float] = {}
    evaluations = 0

    def v(mask: int) -> float:
        nonlocal evaluations
        if mask not in cache:
            cache[mask] = float(game(mask))
            evaluations += 1
        return cache[mask]

    marginals = np.zeros((passes, n))
    n_drawn = 0
    while n_drawn < passes:
        perm = rng.permutation(n)
        variants = [perm, perm[::-1]] if antithetic else [perm]
        for p in variants:
            if n_drawn >= passes:
                break
            mask = 0
            prev = v(0)
            for player in p:
                mask |= 1 << int(player)
                cur = v(mask)
                marginals[n_drawn, player] = cur - prev
                prev = cur
            n_drawn += 1

    phi = marginals.mean(axis=0)
    stderr = marginals.std(axis=0, ddof=1) / np.sqrt(passes)
    return SampledAttribution(
        phi=phi, stderr=stderr, passes=passes, evaluations=evaluations
    )


def resample_pick_stability(
    game: Game,
    n_players: int,
    passes: int,
    k: int,
    n_resamples: int = 50,
    seed: int = 0,
    selector=None,
    antithetic: bool = True,
) -> dict:
    """Measure how often independent re-estimates agree on the selection.

    Runs the sampled estimator ``n_resamples`` times at the given budget,
    applies the selector (default: plain top-k by Shapley value), and
    tabulates how often each coalition mask is picked. A split that persists
    as ``passes`` grows indicates near-substitute players rather than
    sampling noise: the game itself does not distinguish the tied picks.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    passes : int
        Sampling budget (permutations) per re-estimate.
    k : int
        Coalition size handed to the default top-k selector.
    n_resamples : int
        Number of independent re-estimates.
    seed : int
        Base seed; re-estimate ``r`` uses ``seed + r``.
    selector : callable, optional
        Function mapping an attribution vector to a coalition mask;
        defaults to top-k by Shapley value.
    antithetic : bool
        Whether the underlying estimator uses antithetic pairing. Exact
        substitutes tie under pairing, so deterministic tie-breaking masks
        their instability; synthetic probes should pass ``False``.

    Returns
    -------
    dict
        Dictionary with keys ``counts`` (mask -> times picked),
        ``modal_agreement`` (fraction of re-estimates agreeing on the modal
        pick), ``passes``, and ``n_resamples``.
    """
    from topobench.explain.selection import top_k_mask

    pick = selector or (lambda phi: top_k_mask(phi, k))
    counts: dict[int, int] = {}
    for r in range(n_resamples):
        att = sampled_shapley(
            game,
            n_players,
            passes=passes,
            seed=seed + r,
            antithetic=antithetic,
        )
        mask = pick(att.phi)
        counts[mask] = counts.get(mask, 0) + 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "modal_agreement": max(counts.values()) / total,
        "passes": passes,
        "n_resamples": n_resamples,
    }
