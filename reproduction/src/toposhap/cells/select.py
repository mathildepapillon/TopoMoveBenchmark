"""TopoSHAP-Select: the coalition (selection) readout of the cell game.

Per-player Shapley values answer "how much does each cell contribute";
Select answers "which sub-complex is jointly sufficient" — the motif-
extraction task graph explainers are scored on. Both are readouts of
the SAME participation game; on a rank-1 complex with a connectivity
constraint the Select readout recovers SubgraphX's search problem
(whose coalition scoring is the Shapley value of the coalition fused
into one player).

The search is greedy top-down pruning: start from the full complex and
repeatedly detach the cell whose removal least hurts v(S). Top-down
matters: values are evaluated near the training distribution while
choices are being made, whereas bottom-up growth starts from maximally
out-of-distribution singletons and gets trapped (probe: 0.310 vs 0.550
node-GEA on hard benzene graphs). Beam search and coalition-Shapley
re-ranking gave no further gain over greedy pruning (0.771 vs 0.780),
so the fleet uses the cheapest member of the family.
"""

from __future__ import annotations

import numpy as np


def greedy_prune_schedule(game, n_players: int, budgets) -> dict:
    """Greedy top-down prune, recording the coalition at each budget.

    Pruning is nested, so all requested budgets come out of one pass:
    prune from the full complex down to the smallest budget, snapshotting
    S whenever its size hits a requested budget.

    Parameters
    ----------
    game : callable
        Bitmask game: ``game(mask) -> float`` (bit i = player i present).
    n_players : int
        Total number of players.
    budgets : iterable of int
        Coalition sizes to record (values ``>= n_players`` return the
        full set; non-positive values are ignored).

    Returns
    -------
    dict
        ``{budget: (frozenset players, value at that size)}``.
    """
    wanted = sorted({int(b) for b in budgets if int(b) > 0}, reverse=True)
    out: dict[int, tuple[frozenset, float]] = {}
    if not wanted:
        return out

    S = set(range(n_players))
    v_S = game((1 << n_players) - 1)
    for b in [b for b in wanted if b >= n_players]:
        out[b] = (frozenset(S), float(v_S))
    lo = min(wanted)
    while len(S) > lo:
        best_c, best_v = None, -np.inf
        for c in sorted(S):
            v = game(sum(1 << p for p in S - {c}))
            if v > best_v:
                best_c, best_v = c, v
        S.remove(best_c)
        v_S = best_v
        if len(S) in wanted:
            out[len(S)] = (frozenset(S), float(v_S))
    return out
