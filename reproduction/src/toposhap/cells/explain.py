"""Leg 1: cell-native Shapley explanations of individual predictions.

Players are the cells of a combinatorial complex (nodes, edges, faces — any
rank). The game masks the features of absent cells to a baseline (zeros by
default) and reads the model's output for the explained target. Exact values
via :func:`toposhap.shapley.exact.shapley_values` when the player set is small
(GraphXAI motifs), permutation sampling otherwise.

NOTE: the validated original of this module (``topobench.explain``, experiment
#1, exactness vs brute force to 2.254e-14) is ingested at
``data/frozen/reference/exp1_worktree/topobench_repo/topobench/explain/``.
The semantic diff against it is documented in docs/parity-report.md;
tests/test_parity_original.py imports the original directly and checks
numerical parity on the two-triangle fixture, and
tests/test_shapley_exact.py enforces the same exactness bar against an
independent brute force. With ``encoder`` given, this game reproduces the
original's semantics (encoded-feature masking, rank-dependent baselines,
target-class-logit value) by default; the reimplementation's zero/raw
behavior stays available as explicit options.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from toposhap.shapley.exact import shapley_interaction, shapley_values
from toposhap.shapley.games import CachedGame
from toposhap.shapley.sampled import sampled_shapley

#: Practical exact-enumeration ceiling: 2^20 forward passes.
EXACT_PLAYER_LIMIT = 20


@dataclass(frozen=True)
class CellPlayer:
    """One player: cell ``index`` within feature matrix ``x_{rank}``."""

    rank: int
    index: int


@dataclass
class CellExplanation:
    players: list[CellPlayer]
    phi: np.ndarray
    interactions: dict | None  # {frozenset(player positions): value} or None
    exact: bool
    evaluations: int


class CellMaskingGame:
    """v(S) = model output for the target, with cells outside S masked.

    ``model_fn`` maps a batch to the scalar being explained. The ORIGINAL
    validated explainer (``masking.py:MaskedGCCNGame``, the game behind the
    frozen leg-1 numbers) reads the **target-class logit** (``target="label"``
    by default there); pass a ``model_fn`` returning that logit to reproduce
    it. Accuracy-style 0/1 scoring stays available for the neighborhood games
    (experiment #8 default) — the value convention is whatever ``model_fn``
    returns.

    Masking point — raw vs encoded (original semantics):
        With ``encoder=None`` (legacy default) masking acts on the RAW feature
        rows of ``x_{rank}``. The original instead encodes once and masks the
        ENCODED rows — the feature encoder is fixed preprocessing outside the
        game. Pass ``encoder`` (e.g. ``model.feature_encoder``) to reproduce
        that: it is applied to ``batch`` exactly once at construction (under
        ``no_grad``), the game then masks the encoded rows, and ``model_fn``
        must therefore run only the post-encoder stages (backbone + readout),
        never the encoder again.

    Baselines — zeros vs rank-dependent (original semantics):
        ``baseline`` accepts
        * ``None`` — ``"complex_mean"`` when ``encoder`` is given (the
          original's rank-dependent semantics), ``"zeros"`` otherwise (legacy
          default, unchanged);
        * ``"zeros"`` — masked rows become zero vectors;
        * ``"complex_mean"`` — per-rank mean over this complex's own
          (post-encoder) feature rows, the original's self-contained
          rank-dependent baseline;
        * ``dict[rank] -> tensor[hidden]`` — one baseline row per rank,
          broadcast to every masked cell of that rank. This is the original's
          ``baseline_rows`` convention; feed it per-rank train-split means to
          reproduce its ``"train_mean"`` default exactly;
        * ``dict[rank] -> tensor[n_cells, hidden]`` — per-cell replacement
          rows (legacy convention, kept).

    Either way only features are replaced — incidence structure stays intact,
    so the game is exactly "this cell's signal is absent", not "the complex is
    rewired".
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
            # Original semantics: encode once; the game lives on encoded rows.
            with torch.no_grad():
                batch = encoder(batch)
        self.batch = batch
        self.players = players
        # Snapshot EVERY rank's features, not just player ranks: TopoTune's
        # forward overwrites batch.x_{rank} with hidden states, so each
        # evaluation must start from pristine features on all ranks.
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
        self.baseline_name = baseline if isinstance(baseline, str) else "custom"
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
                elif fill.dim() == 1:  # one row per rank (original convention)
                    x[player.index] = fill
                else:  # per-cell rows (legacy convention)
                    x[player.index] = fill[player.index]
            with torch.no_grad():
                return float(self.model_fn(self.batch))
        finally:
            for rank, original in self._originals.items():
                setattr(self.batch, f"x_{rank}", original)


class HopseCellMaskingGame:
    """v(S) for HOPSE models: mask absent cells' encoded per-hop rows.

    HOPSE's wrapper never reads ``x_{rank}`` — it assembles the backbone
    input from the per-hop encoding tensors ``x{rank}_{hop}`` written by
    ``HOPSEFeatureEncoder``. ``CellMaskingGame`` is therefore a structural
    no-op on HOPSE (v(full) == v(empty) exactly; measured on MANTRA,
    2026-08-18). This game applies the original encoded-feature-masking
    semantics at the interface HOPSE actually consumes: encode once, then
    zero the masked cell's row in EVERY hop tensor of its rank.

    Only the ``zeros`` baseline is defined here ("this cell's structural
    encodings are absent"); a complex mean over positional encodings has
    no signal-absent reading.
    """

    def __init__(self, model_fn, batch, players: list[CellPlayer],
                 encoder, max_hop: int):
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
    """Exact when |players| <= EXACT_PLAYER_LIMIT, sampled otherwise.

    ``baseline`` and ``encoder`` are forwarded to :class:`CellMaskingGame`;
    passing ``encoder`` selects the original leg-1 semantics (encoded-feature
    masking with a rank-dependent baseline) by default. Interaction indices
    (pairwise) are only computed in the exact regime — sampled interaction
    estimation is out of scope for this paper.
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
            f"exact interactions need <= {EXACT_PLAYER_LIMIT} players, "
            f"got {n}"
        )
    att = sampled_shapley(game, n, passes=passes, seed=seed)
    return CellExplanation(
        players=players,
        phi=att.phi,
        interactions=None,
        exact=False,
        evaluations=game.calls,
    )
