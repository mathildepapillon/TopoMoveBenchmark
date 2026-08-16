"""Cell-level Shapley explanations of individual predictions.

Players are the cells of a complex (nodes, edges, faces — any rank). The
game masks the features of absent cells to a baseline (zeros by default) and
reads the model's output for the explained target. Values are exact via
:func:`topobench.explain.shapley.shapley_values` when the player set is
small, and permutation-sampled otherwise.
"""

from dataclasses import dataclass

import numpy as np
import torch

from topobench.explain.games import CachedGame
from topobench.explain.shapley import (
    sampled_shapley,
    shapley_interaction,
    shapley_values,
)

#: Practical exact-enumeration ceiling: 2^20 forward passes.
EXACT_PLAYER_LIMIT = 20


@dataclass(frozen=True)
class CellPlayer:
    """One player: the cell ``index`` within feature matrix ``x_{rank}``.

    Parameters
    ----------
    rank : int
        Rank of the cell (0 for nodes, 1 for edges, ...).
    index : int
        Row index of the cell within ``batch.x_{rank}``.
    """

    rank: int
    index: int


@dataclass
class CellExplanation:
    """Shapley explanation of one prediction in terms of cells.

    Parameters
    ----------
    players : list[CellPlayer]
        The explained cells, in player (bit) order.
    phi : np.ndarray
        Shapley value per player.
    interactions : dict, optional
        Pairwise interaction indices ``{frozenset(player positions):
        value}``, or None if not computed.
    exact : bool
        Whether the values are exact (True) or sampled (False).
    evaluations : int
        Number of distinct model evaluations performed.
    """

    players: list[CellPlayer]
    phi: np.ndarray
    interactions: dict | None
    exact: bool
    evaluations: int


class CellMaskingGame:
    """Game whose value is the model output with absent cells masked.

    ``v(S)`` runs the model on the batch with the features of every cell
    outside ``S`` masked, and reads the scalar being explained. Masking
    zeroes the masked cells' rows in ``x_{rank}`` (or replaces them with
    the corresponding ``baseline[rank]`` rows if given) — features only;
    the incidence structure stays intact, so the game is exactly "this
    cell's signal is absent", not "the complex is rewired".

    Parameters
    ----------
    model_fn : callable
        Maps a batch to the scalar being explained (e.g. the logit or
        probability of the predicted class for one sample).
    batch : object
        Batch object exposing per-rank feature matrices ``x_{rank}``.
    players : list[CellPlayer]
        Cells acting as players, in player (bit) order.
    baseline : dict[int, torch.Tensor], optional
        Per-rank replacement features for masked cells; zeros by default.
    """

    def __init__(
        self,
        model_fn,
        batch,
        players: list[CellPlayer],
        baseline: dict[int, torch.Tensor] | None = None,
    ):
        self.model_fn = model_fn
        self.batch = batch
        self.players = players
        self.baseline = baseline or {}
        # Snapshot EVERY rank's features, not just player ranks: backbones
        # such as TopoTune overwrite batch.x_{rank} with hidden states
        # during forward, so each evaluation must start from pristine
        # features on all ranks.
        self._originals = {}
        rank = 0
        while hasattr(batch, f"x_{rank}"):
            self._originals[rank] = getattr(batch, f"x_{rank}").clone()
            rank += 1
        missing = {p.rank for p in players} - set(self._originals)
        if missing:
            raise ValueError(f"batch has no features for ranks {missing}")

    def __call__(self, mask: int) -> float:
        """Evaluate the model with cells outside the coalition masked.

        Parameters
        ----------
        mask : int
            Coalition bitmask over the players.

        Returns
        -------
        float
            The model output for the explained target.
        """
        try:
            for rank, original in self._originals.items():
                setattr(self.batch, f"x_{rank}", original.clone())
            for pos, player in enumerate(self.players):
                if mask >> pos & 1:
                    continue
                x = getattr(self.batch, f"x_{player.rank}")
                fill = self.baseline.get(player.rank)
                x[player.index] = (
                    fill[player.index] if fill is not None else 0.0
                )
            with torch.no_grad():
                return float(self.model_fn(self.batch))
        finally:
            for rank, original in self._originals.items():
                setattr(self.batch, f"x_{rank}", original)


def explain_cells(
    model_fn,
    batch,
    players: list[CellPlayer],
    interactions: bool = False,
    passes: int = 128,
    seed: int | None = None,
) -> CellExplanation:
    """Explain one prediction by Shapley values over cells.

    Exact enumeration is used when ``len(players)`` is at most
    :data:`EXACT_PLAYER_LIMIT`, permutation sampling otherwise. Pairwise
    interaction indices are only computed in the exact regime.

    Parameters
    ----------
    model_fn : callable
        Maps a batch to the scalar being explained.
    batch : object
        Batch object exposing per-rank feature matrices ``x_{rank}``.
    players : list[CellPlayer]
        Cells acting as players, in player (bit) order.
    interactions : bool
        Whether to also compute pairwise interaction indices (exact regime
        only).
    passes : int
        Sampling budget (permutations) used in the sampled regime.
    seed : int, optional
        Seed for the sampled regime.

    Returns
    -------
    CellExplanation
        Per-cell Shapley values, optional interactions, and cost
        accounting.
    """
    n = len(players)
    raw_game = CellMaskingGame(model_fn, batch, players)
    game = CachedGame(n_players=n, evaluate=raw_game)

    if n <= EXACT_PLAYER_LIMIT:
        phi = shapley_values(game, n)
        inter = shapley_interaction(game, n, order=2) if interactions else None
        return CellExplanation(
            players=players,
            phi=phi,
            interactions=inter,
            exact=True,
            evaluations=game.calls,
        )
    if interactions:
        raise ValueError(
            f"exact interactions need <= {EXACT_PLAYER_LIMIT} players, got {n}"
        )
    att = sampled_shapley(game, n, passes=passes, seed=seed)
    return CellExplanation(
        players=players,
        phi=att.phi,
        interactions=None,
        exact=False,
        evaluations=game.calls,
    )
