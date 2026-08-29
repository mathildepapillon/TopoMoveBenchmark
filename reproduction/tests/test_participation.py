"""Participation game: detachment semantics, fidelity guard, conduits."""

import numpy as np
import torch

from toposhap.cells import CellPlayer
from toposhap.cells.explain import CellMaskingGame
from toposhap.cells.participation import CellParticipationGame
from toposhap.neighborhoods import CoalitionMaskedBackbone, prune_backbone_
from toposhap.shapley.exact import shapley_values
from toposhap.shapley.games import CachedGame
from toposhap.testing import (
    full_vocabulary_backbone,
    two_triangle_batch,
    two_triangle_participation_batch,
)

N0, N1, N2 = 4, 5, 2
ALL_PLAYERS = (
    [CellPlayer(rank=0, index=i) for i in range(N0)]
    + [CellPlayer(rank=1, index=i) for i in range(N1)]
    + [CellPlayer(rank=2, index=i) for i in range(N2)]
)
FULL = (1 << len(ALL_PLAYERS)) - 1


def _model_fn(backbone):
    def fn(batch):
        with torch.no_grad():
            return backbone(batch)[0].sum()
    return fn


def _games(mask_bits):
    backbone = full_vocabulary_backbone(seed=6)
    prune_backbone_(backbone, mask_bits)
    fn = _model_fn(backbone)
    part = CellParticipationGame(
        fn, two_triangle_participation_batch(seed=7), ALL_PLAYERS)
    sig = CellMaskingGame(
        fn, two_triangle_participation_batch(seed=7), ALL_PLAYERS)
    return fn, part, sig


def test_full_coalition_matches_pristine():
    backbone = full_vocabulary_backbone(seed=6)
    fn = _model_fn(backbone)
    pristine = float(fn(two_triangle_participation_batch(seed=7)))
    game = CellParticipationGame(
        fn, two_triangle_participation_batch(seed=7), ALL_PLAYERS)
    assert abs(game(FULL) - pristine) < 1e-6


def test_efficiency_axiom_exact():
    _, part, _ = _games(1024)  # single 2-down_incidence-2 route
    game = CachedGame(n_players=len(ALL_PLAYERS), evaluate=part)
    phi = shapley_values(game, len(ALL_PLAYERS))
    assert abs(sum(phi) - (game(FULL) - game(0))) < 1e-5


def test_conduit_edges_and_pool_vertices_earn_credit():
    """Route 2->0 (faces->nodes through edges): the signal game zeroes
    edges and vertices; participation credits both roles."""
    _, part, sig = _games(1024)
    gp = CachedGame(n_players=len(ALL_PLAYERS), evaluate=part)
    gs = CachedGame(n_players=len(ALL_PLAYERS), evaluate=sig)
    phi_p = np.array(shapley_values(gp, len(ALL_PLAYERS)))
    phi_s = np.array(shapley_values(gs, len(ALL_PLAYERS)))
    edges = slice(N0, N0 + N1)
    verts = slice(0, N0)
    assert np.abs(phi_s[edges]).max() < 1e-9, "signal game: edges null"
    assert np.abs(phi_s[verts]).max() < 1e-9, "signal game: vertices null"
    assert np.abs(phi_p[edges]).max() > 1e-4, "participation: conduit credit"
    assert np.abs(phi_p[verts]).max() > 1e-4, "participation: pool credit"


def test_output_invariant_ranks_stay_null():
    """CWN-style wiring (all routes end at rank 1) with a rank-0 readout:
    edges and faces are provably output-invariant, so even the
    participation game must give them exactly zero."""
    _, part, _ = _games(26)
    game = CachedGame(n_players=len(ALL_PLAYERS), evaluate=part)
    phi = np.array(shapley_values(game, len(ALL_PLAYERS)))
    assert np.abs(phi[N0:]).max() < 1e-9
    assert np.abs(phi[:N0]).max() > 1e-4


def test_strict_guard_rejects_convention_mismatch():
    backbone = full_vocabulary_backbone(seed=6)
    fn = _model_fn(backbone)
    batch = two_triangle_participation_batch(seed=7)
    bad = batch.incidence_1.coalesce()
    batch.incidence_1 = torch.sparse_coo_tensor(
        bad.indices()[:, :-1], bad.values()[:-1], bad.size())
    import pytest
    with pytest.raises(ValueError, match="rebuild"):
        CellParticipationGame(fn, batch, ALL_PLAYERS)


class _GhostBackbone(torch.nn.Module):
    """Biased rank-0 update: every vertex row gains +1, detached or not."""

    def forward(self, batch):
        batch.x_0 = batch.x_0 + 1.0
        return {"x_0": batch.x_0, "x_1": batch.x_1, "x_2": batch.x_2}


class _SumReadout:
    pooling_type = "sum"

    def forward(self, model_out, batch):
        return model_out

    def compute_logits(self, x, batch_idx):
        s = x.sum(0, keepdim=True).sum(-1, keepdim=True)
        return torch.cat([s, -s], dim=1)


class _GhostModel:
    feature_encoder = staticmethod(lambda b: b)
    backbone = _GhostBackbone()
    readout = _SumReadout()


def test_pool_faithful_margin_drops_ghost_rows():
    """A biased stage writing to x_0 gives detached vertices nonzero
    "ghost" rows; sum pooling must not credit them. pool_faithful_margin
    zeroes detached rows pre-pool, matching a physical row drop."""
    from toposhap.cells.participation import pool_faithful_margin

    model = _GhostModel()
    game = CellParticipationGame(
        pool_faithful_margin(model, 0, 1), two_triangle_participation_batch(seed=7),
        ALL_PLAYERS)
    missing_v0 = FULL & ~1  # detach vertex 0 only
    got = game(missing_v0)
    x0 = two_triangle_participation_batch(seed=7).x_0
    expected = float(2 * (x0[1:] + 1.0).sum())  # present rows, physically
    assert abs(got - expected) < 1e-5
    assert not hasattr(game.batch, "toposhap_absent"), "cleanup"

    def naive_fn(b):
        o = model.backbone(b)
        o = model.readout.forward(model_out=o, batch=b)
        logits = model.readout.compute_logits(o["x_0"], None)
        return float(logits[0, 0] - logits[0, 1])

    naive = CellParticipationGame(
        naive_fn, two_triangle_participation_batch(seed=7), ALL_PLAYERS)
    leak = abs(naive(missing_v0) - expected)
    assert leak > 0.5, f"probe should reproduce the leak, got {leak}"


def test_deletion_curve_and_auc():
    """Deletion under participation removal: endpoints match v(full)
    and v(empty); the Shapley ranking beats random on AUC."""
    from toposhap.baselines import occlusion_scores, random_scores
    from toposhap.metrics.deletion import deletion_auc, deletion_curve

    backbone = full_vocabulary_backbone(seed=6)
    prune_backbone_(backbone, 1024)
    fn = _model_fn(backbone)
    game_raw = CellParticipationGame(
        fn, two_triangle_participation_batch(seed=7), ALL_PLAYERS)
    game = CachedGame(n_players=len(ALL_PLAYERS), evaluate=game_raw)
    n = len(ALL_PLAYERS)

    phi = shapley_values(game, n)
    v_full = game((1 << n) - 1)
    sgn = 1.0 if v_full >= 0 else -1.0
    rank_phi = np.argsort(sgn * np.asarray(phi))[::-1]
    curve_phi = deletion_curve(game, rank_phi, n)
    assert len(curve_phi) == n + 1
    assert abs(curve_phi[0] - v_full) < 1e-9
    assert abs(curve_phi[-1] - game(0)) < 1e-9

    occ = occlusion_scores(game, n)
    assert occ.shape == (n,)
    rnd_aucs = []
    for s in range(5):
        rank_r = np.argsort(random_scores(n, seed=s))[::-1]
        rnd_aucs.append(deletion_auc(deletion_curve(game, rank_r, n),
                                     v_empty=game(0), orient_sign=sgn))
    auc_phi = deletion_auc(curve_phi, v_empty=game(0), orient_sign=sgn)
    assert auc_phi <= float(np.mean(rnd_aucs)) + 1e-9, \
        (auc_phi, float(np.mean(rnd_aucs)))


def test_gradient_x_input_shapes_and_nulls():
    """gradient x input: one score per player; ranks the readout
    provably never reads get exactly zero."""
    import types

    from toposhap.baselines import gradient_x_input

    class _Readout:
        pooling_type = "sum"

        def forward(self, model_out, batch):
            return model_out

        def compute_logits(self, x, batch_idx):
            s = x.sum(0, keepdim=True).sum(-1, keepdim=True)
            return torch.cat([s, -s], dim=1)

    model = types.SimpleNamespace(
        feature_encoder=lambda b: b,
        backbone=lambda b: {"x_0": b.x_0, "x_1": b.x_1, "x_2": b.x_2},
        readout=_Readout())
    batch = two_triangle_participation_batch(seed=7)
    scores = gradient_x_input(model, batch, ALL_PLAYERS, 0, 1)
    assert scores.shape == (len(ALL_PLAYERS),)
    assert np.abs(scores[N0:]).max() == 0.0
    assert np.abs(scores[:N0]).max() > 0.0
