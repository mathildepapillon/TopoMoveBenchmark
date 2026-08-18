"""Cell-level Shapley explanations of individual predictions.

Players are the cells of a complex (nodes, edges, faces — any rank). The
game masks the features of absent cells to a baseline and reads the model's
output for the explained target. Masking can act on the raw feature rows
(default) or, when an ``encoder`` is supplied, on the encoded rows the
message-passing backbone consumes — the encoder is then applied once and
treated as fixed preprocessing outside the game. Values are exact via
:func:`topobench.explain.shapley.shapley_values` when the player set is
small, and permutation-sampled otherwise.

Masking must reach the tensors the model actually reads. Models that
consume per-hop encodings (HOPSE) assemble their backbone input from the
tensors ``x{rank}_{hop}`` and never read ``x_{rank}``, so
:class:`CellMaskingGame` is a structural no-op on them — use
:class:`HopseCellMaskingGame` instead, which masks the encoded per-hop
rows.
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
    outside ``S`` masked, and reads the scalar being explained (for logit
    games, typically the target-class logit). Only features are replaced;
    the incidence structure stays intact, so the game is exactly "this
    cell's signal is absent", not "the complex is rewired".

    Parameters
    ----------
    model_fn : callable
        Maps a batch to the scalar being explained (e.g. the logit or
        probability of the predicted class for one sample). When
        ``encoder`` is given, ``model_fn`` must run only the post-encoder
        stages (backbone and readout), never the encoder again.
    batch : object
        Batch object exposing per-rank feature matrices ``x_{rank}``.
    players : list[CellPlayer]
        Cells acting as players, in player (bit) order.
    baseline : dict[int, torch.Tensor] or str, optional
        What masked rows become. ``"zeros"`` (default without an encoder)
        fills with zero vectors; ``"complex_mean"`` (default with an
        encoder) uses the per-rank mean over this complex's own
        (post-encoder) feature rows; a dict maps rank to either one
        baseline row of shape ``[hidden]`` (broadcast to every masked cell
        of that rank, e.g. per-rank train-split means) or a per-cell
        replacement matrix of shape ``[n_cells, hidden]``.
    encoder : callable, optional
        Feature encoder applied to ``batch`` exactly once at construction
        (under ``no_grad``). Masking then acts on the encoded rows, making
        the encoder fixed preprocessing outside the game.
    """

    def __init__(
        self,
        model_fn,
        batch,
        players: list[CellPlayer],
        baseline: dict[int, torch.Tensor] | str | None = None,
        encoder=None,
    ):
        self.model_fn = model_fn
        self.encoder = encoder
        if encoder is not None:
            # Encode once; the game lives on encoded rows.
            with torch.no_grad():
                batch = encoder(batch)
        self.batch = batch
        self.players = players
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

        if baseline is None:
            baseline = "complex_mean" if encoder is not None else "zeros"
        self.baseline_name = (
            baseline if isinstance(baseline, str) else "custom"
        )
        player_ranks = {p.rank for p in players}
        if baseline == "zeros":
            self.baseline = {}
        elif baseline == "complex_mean":
            self.baseline = {
                r: self._originals[r].mean(0) for r in player_ranks
            }
        elif isinstance(baseline, dict):
            self.baseline = {
                r: fill.to(self._originals[r].dtype)
                for r, fill in baseline.items()
            }
        else:
            raise ValueError(f"unknown baseline {baseline!r}")

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
                if fill is None:
                    x[player.index] = 0.0
                elif fill.dim() == 1:  # one baseline row per rank
                    x[player.index] = fill
                else:  # per-cell replacement rows
                    x[player.index] = fill[player.index]
            with torch.no_grad():
                return float(self.model_fn(self.batch))
        finally:
            for rank, original in self._originals.items():
                setattr(self.batch, f"x_{rank}", original)


class HopseCellMaskingGame:
    """Cell-masking game for models that consume per-hop encodings (HOPSE).

    HOPSE-style models never read ``x_{rank}``: the wrapper assembles the
    backbone input from the per-hop encoding tensors ``x{rank}_{hop}``
    written by the feature encoder (see
    :class:`~topobench.nn.encoders.hopse_encoder.HOPSEFeatureEncoder`).
    :class:`CellMaskingGame` is therefore a structural no-op on such
    models — masking ``x_{rank}`` never reaches the computation, so
    ``v(full) == v(empty)`` exactly and every Shapley value is 0. This
    game applies the encoded-feature-masking semantics at the interface
    the model actually consumes: the encoder runs exactly once at
    construction, and the game then zeroes the masked cell's row in EVERY
    hop tensor of its rank.

    Only the ``"zeros"`` baseline is defined here ("this cell's
    structural encodings are absent"); a complex mean over positional
    encodings has no signal-absent reading.

    Parameters
    ----------
    model_fn : callable
        Maps a batch to the scalar being explained. Must run only the
        post-encoder stages (backbone and readout), never the encoder
        again.
    batch : object
        Batch object; after encoding it must expose one tensor
        ``x{rank}_{hop}`` per player rank and hop.
    players : list[CellPlayer]
        Cells acting as players, in player (bit) order.
    encoder : callable
        Feature encoder applied to ``batch`` exactly once at construction
        (under ``no_grad``); it writes the per-hop tensors the game masks.
    max_hop : int
        Number of hop tensors per rank
        (``x{rank}_0 ... x{rank}_{max_hop - 1}``).
    """

    def __init__(
        self,
        model_fn,
        batch,
        players: list[CellPlayer],
        encoder,
        max_hop: int,
    ):
        self.model_fn = model_fn
        self.players = players
        self.max_hop = max_hop
        self.baseline_name = "zeros"
        with torch.no_grad():
            batch = encoder(batch)
        self.batch = batch
        self._originals = {}
        for rank in sorted({p.rank for p in players}):
            for hop in range(max_hop):
                key = f"x{rank}_{hop}"
                if not hasattr(batch, key):
                    raise ValueError(f"batch has no encoded tensor {key}")
                self._originals[key] = getattr(batch, key).clone()

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
            for key, original in self._originals.items():
                setattr(self.batch, key, original.clone())
            for pos, player in enumerate(self.players):
                if mask >> pos & 1:
                    continue
                for hop in range(self.max_hop):
                    x = getattr(self.batch, f"x{player.rank}_{hop}")
                    x[player.index] = 0.0
            with torch.no_grad():
                return float(self.model_fn(self.batch))
        finally:
            for key, original in self._originals.items():
                setattr(self.batch, key, original)


def explain_cells(
    model_fn,
    batch,
    players: list[CellPlayer],
    interactions: bool = False,
    passes: int = 128,
    seed: int | None = None,
    baseline: dict[int, torch.Tensor] | str | None = None,
    encoder=None,
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
    baseline : dict[int, torch.Tensor] or str, optional
        Forwarded to :class:`CellMaskingGame`.
    encoder : callable, optional
        Forwarded to :class:`CellMaskingGame`; when given, masking acts on
        encoded features and ``model_fn`` must not re-apply the encoder.

    Returns
    -------
    CellExplanation
        Per-cell Shapley values, optional interactions, and cost
        accounting.
    """
    n = len(players)
    raw_game = CellMaskingGame(
        model_fn, batch, players, baseline=baseline, encoder=encoder
    )
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
