"""Deletion curves under participation removal.

The removal operator is DECLARED: detaching a cell means the
participation semantics of :mod:`toposhap.cells.participation` —
features zeroed, wiring rebuilt without the cell, readout pooling
excluding it. A deletion curve removes cells in a ranking's order,
most-important first, and records the model output after each removal;
a faithful ranking makes the output collapse toward v(empty) fastest.

All evaluations go through the caller's game callable, so cached
coalitions (e.g. from a Shapley run on the same game) are reused.
"""

from __future__ import annotations

import numpy as np

__all__ = ["deletion_curve", "deletion_auc"]


def deletion_curve(game, ranking, n_players: int) -> list[float]:
    """v after cumulatively detaching players in ``ranking`` order.

    Parameters
    ----------
    game : callable
        mask (int bitmask of PRESENT players) -> float value.
    ranking : sequence of int
        Player indices, most important first. May be a prefix; the
        curve stops when the ranking is exhausted.
    n_players : int
        Total number of players.

    Returns
    -------
    list of length len(ranking)+1: [v(full), v(full minus top-1), ...].
    """
    full = (1 << n_players) - 1
    mask = full
    vals = [float(game(full))]
    for i in ranking:
        mask &= ~(1 << int(i))
        vals.append(float(game(mask)))
    return vals


def deletion_auc(vals, v_empty: float | None = None,
                 orient_sign: float = 1.0,
                 normalize: bool = True) -> float:
    """Normalized area under the deletion curve; lower = more faithful.

    The curve is oriented so that the full-coalition value maps to 1
    and ``v_empty`` (default: the curve's final value) maps to 0:
    a ranking that destroys the prediction immediately scores ~0, a
    ranking that never moves it scores ~1. ``orient_sign`` flips the
    margin first when the explained quantity can be negative.
    """
    v = np.asarray(vals, dtype=float) * orient_sign
    if not normalize:
        # plain area under the curve (standard XAI deletion AUC);
        # bounded when vals is bounded (e.g. class probabilities)
        x = np.linspace(0.0, 1.0, len(v))
        trap = getattr(np, 'trapezoid', np.trapz)
        return float(trap(v, x))
    v0 = v[0]
    ve = v[-1] if v_empty is None else float(v_empty) * orient_sign
    span = v0 - ve
    if abs(span) < 1e-12:
        return 1.0  # degenerate: removal never changes the output
    unit = (v - ve) / span
    x = np.linspace(0.0, 1.0, len(v))
    trap = getattr(np, 'trapezoid', np.trapz)
    return float(trap(unit, x))
