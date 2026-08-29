"""Explanation-quality metrics for leg 1: GEF faithfulness and motif GEA.

Conventions match the GraphXAI benchmark suite (Agarwal et al.) so our numbers
are comparable to the SubgraphX/GNNExplainer/PGExplainer baselines:

* GEF (unfaithfulness): 1 - exp(-KL(f(complex) || f(complex masked to the
  explanation))). Lower is better. Reported with multi-seed error bars; the
  GEF advantage survives all controls (experiment #4).
* GEA (accuracy): Jaccard agreement between the explanation's chosen cells
  and the ground-truth motif cells, top-k'd to the motif size.
"""

from __future__ import annotations

import numpy as np


def gef_unfaithfulness(p_full: np.ndarray, p_masked: np.ndarray) -> float:
    """1 - exp(-KL(p_full || p_masked)) over class distributions.

    Both inputs are probability vectors (softmax outputs). 0 = perfectly
    faithful explanation, 1 = maximally unfaithful.
    """
    p = np.clip(np.asarray(p_full, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(p_masked, dtype=float), 1e-12, 1.0)
    p, q = p / p.sum(), q / q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return 1.0 - float(np.exp(-kl))


def gea_jaccard(explanation_cells: set, motif_cells: set) -> float:
    """Jaccard(explanation, ground-truth motif). 1 = exact motif recovery."""
    if not explanation_cells and not motif_cells:
        return 1.0
    inter = len(explanation_cells & motif_cells)
    union = len(explanation_cells | motif_cells)
    return inter / union


def top_cells_by_attribution(
    phi: np.ndarray, players: list, k: int
) -> set:
    """The k highest-attributed players, as a set (for GEA)."""
    order = np.argsort(-phi, kind="stable")[:k]
    return {players[i] for i in order}


def random_control_gea(
    players: list, motif_cells: set, k: int, n_draws: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Mean and sd of GEA under uniformly random k-subsets — the floor every
    explainer must beat for its GEA to mean anything."""
    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(len(players))
    for _ in range(n_draws):
        chosen = {players[i] for i in rng.choice(idx, size=k, replace=False)}
        scores.append(gea_jaccard(chosen, motif_cells))
    return float(np.mean(scores)), float(np.std(scores))
