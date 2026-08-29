"""Axioms as executable checks — the trustworthiness machinery.

Every attribution run can be audited against the Shapley axioms it claims.
These checks are cheap relative to the game evaluations they audit, and they
have teeth: they surfaced all three TopoBench framework bugs (a dead route is
a null player that upstream configs said should matter) and the cross-rank
batching bug (via the invariance guard in toposhap.patches).
"""

from __future__ import annotations

import numpy as np

from toposhap.shapley.games import Game


def check_efficiency(
    phi: np.ndarray, game: Game, n_players: int, atol: float = 1e-10
) -> float:
    """sum phi_i must equal v(N) - v(empty). Returns the gap; raises if > atol.

    For sampled attributions pass the estimator's tolerance instead of the
    exact default.
    """
    full = (1 << n_players) - 1
    gap = float(abs(phi.sum() - (game(full) - game(0))))
    if gap > atol:
        raise AssertionError(f"efficiency violated: gap {gap:.3e} > {atol:.1e}")
    return gap


def find_null_players(
    game: Game, n_players: int, atol: float = 1e-10, n_probes: int | None = None
) -> list[int]:
    """Players whose marginal contribution is ~0 on every probed coalition.

    Exhaustive when n_probes is None (2^(n-1) coalitions per player),
    otherwise probes that many random coalitions per player. A route that the
    architecture claims is live showing up here is a bug signal — this is the
    check that exposed the dead inter-rank routes.
    """
    rng = np.random.default_rng(0)
    nulls = []
    for i in range(n_players):
        bit = 1 << i
        if n_probes is None:
            others = [m for m in range(1 << n_players) if not m & bit]
        else:
            others = [
                int(rng.integers(0, 1 << n_players)) & ~bit
                for _ in range(n_probes)
            ]
        if all(abs(game(m | bit) - game(m)) <= atol for m in others):
            nulls.append(i)
    return nulls


def check_symmetry_pair(
    game: Game, n_players: int, i: int, j: int, atol: float = 1e-10
) -> float:
    """Max |v(S+i) - v(S+j)| over all S excluding both — 0 iff i,j symmetric.

    Symmetric players must receive equal Shapley values; substitutes that are
    symmetric in the game but receive unequal sampled attributions indicate
    estimator noise, not model structure.
    """
    bi, bj = 1 << i, 1 << j
    worst = 0.0
    for m in range(1 << n_players):
        if m & (bi | bj):
            continue
        worst = max(worst, abs(game(m | bi) - game(m | bj)))
    return worst
