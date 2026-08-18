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

Masking must also actually change the tensors: when per-rank features are
constant across cells, the ``"complex_mean"`` baseline replaces masked
rows with themselves, so every coalition gets the same value and all
Shapley values are exactly 0. :class:`CellMaskingGame` refuses such
degenerate baselines at construction; ``baseline="zeros"`` is the safe
choice for constant features.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from topobench.explain.games import CachedGame
from topobench.explain.shapley import (
    sampled_shapley,
    shapley_interaction,
    shapley_values,
)

#: Hard exact-enumeration ceiling: never enumerate beyond 2^20 coalitions.
EXACT_PLAYER_LIMIT = 20

#: Default cap on exact-regime model evaluations (2^16); exceeding it
#: falls back to permutation sampling with a warning. See
#: :func:`explain_cells`.
DEFAULT_MAX_EXACT_EVALUATIONS = 65536


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
        replacement matrix of shape ``[n_cells, hidden]``. Note that
        ``"complex_mean"`` degenerates when a rank's features are
        constant across cells: the mean IS every row, so masking that
        rank replaces rows with themselves. ``baseline="zeros"`` is the
        safe choice for constant features.
    encoder : callable, optional
        Feature encoder applied to ``batch`` exactly once at construction
        (under ``no_grad``). Masking then acts on the encoded rows, making
        the encoder fixed preprocessing outside the game.

    Raises
    ------
    ValueError
        At construction, when the setup cannot yield a meaningful game:
        the batch exposes no ``x_{rank}`` feature matrices at all, or a
        player references a rank the batch has no features for; a custom
        baseline dict names ranks that are not player ranks, or an entry's
        shape does not match the rank's feature width; or the resolved
        baseline rows equal the original feature rows on every player
        rank — the game would then be constant (every coalition identical,
        all Shapley values exactly 0).
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
        # features on all ranks. Ranks are discovered both by counting up
        # from x_0 and from the players' own ranks, so a batch without
        # rank-0 features (e.g. edge-only features) is still handled.
        ranks = set()
        rank = 0
        while hasattr(batch, f"x_{rank}"):
            ranks.add(rank)
            rank += 1
        ranks.update(p.rank for p in players if hasattr(batch, f"x_{p.rank}"))
        self._originals = {
            r: getattr(batch, f"x_{r}").clone() for r in sorted(ranks)
        }
        if not self._originals:
            raise ValueError(
                "batch has no x_{rank} feature matrices at all: no x_0, "
                "x_1, ... attribute exists. CellMaskingGame masks per-rank "
                "feature rows, so the batch must expose them"
            )
        missing = {p.rank for p in players} - set(self._originals)
        if missing:
            raise ValueError(
                f"batch has no features for ranks {sorted(missing)}; "
                f"available ranks: {sorted(self._originals)}"
            )

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
            unknown = set(baseline) - player_ranks
            if unknown:
                raise ValueError(
                    f"baseline has entries for ranks {sorted(unknown)}, "
                    "which are not player ranks; valid ranks are "
                    f"{sorted(player_ranks)}"
                )
            self.baseline = {
                r: fill.to(self._originals[r].dtype)
                for r, fill in baseline.items()
            }
            self._validate_baseline_shapes()
        else:
            raise ValueError(f"unknown baseline {baseline!r}")

        # Guard against a degenerate game: if the resolved baseline rows
        # equal the original rows on every player rank (e.g.
        # "complex_mean" when a rank's features are constant, so the mean
        # IS every row), masking replaces rows with themselves — every
        # coalition gets the same value and all Shapley values are
        # exactly 0.
        if all(self._baseline_equals_originals(r) for r in player_ranks):
            raise ValueError(
                f"degenerate baseline {self.baseline_name!r}: the resolved "
                "baseline rows equal the original feature rows on every "
                "player rank, so masking is a no-op — every coalition gets "
                "the same value and all Shapley values are exactly 0 (this "
                "happens e.g. with baseline='complex_mean' when per-rank "
                "features are constant); pass a baseline that differs from "
                'the features, such as baseline="zeros"'
            )

    def _validate_baseline_shapes(self) -> None:
        """Validate custom baseline entries against the feature shapes.

        Raises
        ------
        ValueError
            If a per-rank baseline row does not match the rank's feature
            width, a per-cell replacement matrix does not match the
            rank's feature shape, or an entry has an unsupported number
            of dimensions.
        """
        for r, fill in self.baseline.items():
            original = self._originals[r]
            if fill.dim() == 1:
                if fill.shape[0] != original.shape[1]:
                    raise ValueError(
                        f"baseline row for rank {r} has width "
                        f"{fill.shape[0]}, but rank-{r} features have "
                        f"width {original.shape[1]}"
                    )
            elif fill.dim() == 2:
                if fill.shape != original.shape:
                    raise ValueError(
                        f"per-cell baseline for rank {r} has shape "
                        f"{tuple(fill.shape)}, but rank-{r} features "
                        f"have shape {tuple(original.shape)}"
                    )
            else:
                raise ValueError(
                    f"baseline for rank {r} must be one row of shape "
                    f"[{original.shape[1]}] or a per-cell matrix of shape "
                    f"{tuple(original.shape)}, got a {fill.dim()}-d tensor"
                )

    def _baseline_equals_originals(self, rank: int) -> bool:
        """Check whether masking rows of ``rank`` would replace them with
        themselves.

        Parameters
        ----------
        rank : int
            Player rank to resolve the baseline for.

        Returns
        -------
        bool
            True when the resolved replacement rows equal the original
            feature rows of ``rank`` exactly.
        """
        original = self._originals[rank]
        fill = self.baseline.get(rank)
        if fill is None:  # zeros
            rows = torch.zeros_like(original)
        elif fill.dim() == 1:  # one baseline row per rank
            if fill.shape != original.shape[1:]:
                return False
            rows = fill.expand_as(original)
        else:  # per-cell replacement rows
            if fill.shape != original.shape:
                return False
            rows = fill
        return torch.equal(rows, original)

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
    max_exact_evaluations: int = DEFAULT_MAX_EXACT_EVALUATIONS,
) -> CellExplanation:
    """Explain one prediction by Shapley values over cells.

    Exact enumeration is used when ``len(players)`` is at most
    :data:`EXACT_PLAYER_LIMIT` and the ``2 ** len(players)`` model
    evaluations it costs fit within ``max_exact_evaluations``; permutation
    sampling (budgeted by ``passes``) is used otherwise. Falling back
    because of the evaluation cap emits a warning. Pairwise interaction
    indices are only computed in the exact regime.

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
    max_exact_evaluations : int
        Cap on the number of model evaluations the exact regime may cost
        (default :data:`DEFAULT_MAX_EXACT_EVALUATIONS` = 2^16). When
        ``2 ** len(players)`` exceeds it, the sampled estimator is used
        instead and a warning explains how to restore exactness.

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

    # Constant-game diagnostic: if masking every player changes nothing,
    # the game is (as far as these two coalitions show) constant and all
    # Shapley values will be 0. Both evaluations land in the cache, so
    # the estimators below reuse them for free.
    full_mask = (1 << n) - 1
    if game(full_mask) == game(0):
        warnings.warn(
            "v(full) == v(empty) exactly: masking every player did not "
            "change the model output, so the game looks constant and all "
            "Shapley values will be 0. Known cause: the model may not "
            "read x_{rank} — per-hop-encoding models (HOPSE) need "
            "HopseCellMaskingGame, which masks the x{rank}_{hop} tensors "
            "they actually consume. (A genuinely null model also produces "
            "this, which is why it is a warning, not an error.)",
            stacklevel=2,
        )

    if n <= EXACT_PLAYER_LIMIT and (1 << n) <= max_exact_evaluations:
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
            "exact interactions need the exact regime: at most "
            f"{EXACT_PLAYER_LIMIT} players and 2**n_players <= "
            f"max_exact_evaluations; got {n} players with "
            f"max_exact_evaluations={max_exact_evaluations}"
        )
    if n <= EXACT_PLAYER_LIMIT:
        warnings.warn(
            f"exact Shapley values over {n} players would cost 2**{n} = "
            f"{1 << n} model evaluations, above max_exact_evaluations="
            f"{max_exact_evaluations}; falling back to permutation "
            f"sampling with passes={passes} (approximate values). Pass "
            f"max_exact_evaluations={1 << n} or higher to restore "
            "exactness, or reduce the number of players (or passes) to "
            "reduce cost.",
            stacklevel=2,
        )
    att = sampled_shapley(game, n, passes=passes, seed=seed)
    return CellExplanation(
        players=players,
        phi=att.phi,
        interactions=None,
        exact=False,
        evaluations=game.calls,
    )
