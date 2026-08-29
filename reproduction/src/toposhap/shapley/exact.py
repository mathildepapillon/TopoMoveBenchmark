"""Exact Shapley values and Shapley-interaction indices for small player sets.

Enumeration is over all 2^n coalitions, so this is exact by construction and
tractable for the games in this paper (n = 11 neighborhoods, or the cell games
of leg 1 after restriction to <= ~20 players). The original module's exactness
bar — max error 2.254e-14 vs brute force — is enforced in tests here the same
way: an independent brute-force implementation lives in the test suite, never
in the library, so the two cannot drift together.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from toposhap.shapley.games import Game


def _shapley_weights(n: int) -> np.ndarray:
    """w[s] = s! (n-s-1)! / n! — weight of a marginal from a size-s coalition."""
    return np.array(
        [
            math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            for s in range(n)
        ]
    )


def shapley_values(game: Game, n_players: int) -> np.ndarray:
    """Exact Shapley values phi_i = sum_S w(|S|) [v(S+i) - v(S)].

    Evaluates the game on all 2^n masks exactly once each.
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


def shapley_interaction(game: Game, n_players: int, order: int = 2) -> dict:
    """Exact Shapley interaction index (Grabisch & Roubens) for all subsets
    T of the given order.

    Returns {frozenset(T): I(T)} with
    I(T) = sum_{S ⊆ N\\T} w_{|T|}(|S|) * delta_T v(S),
    where delta_T v(S) is the discrete derivative
    sum_{L ⊆ T} (-1)^{|T|-|L|} v(S ∪ L) and
    w_t(s) = s! (n-s-t)! / (n-t+1)!.
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
    """|sum_i phi_i - (v(N) - v(empty))| — zero for exact Shapley values.

    This is one of the executable axiom checks (see toposhap.metrics.axioms);
    exposed here because estimators report it as a convergence diagnostic.
    """
    full = (1 << n_players) - 1
    return float(abs(phi.sum() - (game(full) - game(0))))
