"""Attribution baselines over the SAME player set as TopoSHAP.

Every baseline returns one score per player (cells, in player order) so
that rankings are directly comparable to Shapley attributions under the
same deletion-curve evaluation (:mod:`toposhap.metrics.deletion`).

- random_scores:     seeded noise (the floor every method must beat)
- occlusion_scores:  v(full) - v(full minus {i}) via the caller's game —
                     leave-one-out under participation removal
- gradient_x_input:  d(margin)/d(x_cell) · x_cell on the RAW features,
                     summed per cell row (GCCN-family models; HOPSE's
                     recomputed encodings put it out of scope here)
"""

from __future__ import annotations

import copy

import numpy as np
import torch

__all__ = ["random_scores", "occlusion_scores", "gradient_x_input"]


def random_scores(n_players: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n_players)


def occlusion_scores(game, n_players: int) -> np.ndarray:
    """Leave-one-out drop per player, through the (cached) game."""
    full = (1 << n_players) - 1
    v_full = float(game(full))
    return np.array([v_full - float(game(full & ~(1 << i)))
                     for i in range(n_players)])


def gradient_x_input(model, batch, players, pos_cls: int,
                     neg_cls: int) -> np.ndarray:
    """grad(margin) . input, summed per cell row, on raw features.

    Runs the full model (encoder + backbone + readout margin) on a
    deep-copied batch with gradients enabled on every ``x_{rank}``.
    """
    b = copy.deepcopy(batch)
    leaves = {}
    ranks = sorted({p.rank for p in players})
    for r in ranks:
        x = getattr(b, f"x_{r}").detach().clone().requires_grad_(True)
        setattr(b, f"x_{r}", x)
        leaves[r] = x
    b = model.feature_encoder(b)
    out = model.backbone(b)
    out = model.readout.forward(model_out=out, batch=b)
    logits = model.readout.compute_logits(
        out["x_0"], getattr(b, "batch_0", None))
    margin = logits[0, pos_cls] - logits[0, neg_cls]
    margin.backward()
    scores = []
    for p in players:
        x = leaves[p.rank]
        g = x.grad
        s = 0.0 if g is None else float(
            (g[p.index] * x[p.index].detach()).sum())
        scores.append(s)
    return np.array(scores)
