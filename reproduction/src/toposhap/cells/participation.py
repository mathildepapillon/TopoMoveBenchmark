"""Participation-semantics cell game: absent cells are detached, not muted.

The frozen leg-1 game (:mod:`toposhap.cells.explain`) answers "whose
*signal* matters": masking replaces a cell's features and leaves the
wiring intact, so receivers, conduits, and pool members earn nothing.
:class:`CellParticipationGame` answers the broader question — which cells
*participate* in the prediction:

    v(S) = the model's output on the complex in which only the cells in
    S participate. A cell outside S is detached everywhere it appears:
    its feature row is zeroed, its entries in every neighborhood matrix
    are removed (as source, target, and intermediate — composed r-hop
    neighborhoods are rebuilt from the masked incidences), and it drops
    out of readout pooling.

This is the cell-complex generalization of induced-subgraph coalition
semantics (SubgraphX-style) from graph XAI. Under it a conduit edge on a
composed triangle-to-vertex route earns credit, and a receiver vertex
earns pool credit — wherever detaching it actually changes the output.
Cells whose presence provably never affects the output (e.g. every edge
of the shared-head CWN expression) remain exact null players, as the
axiom requires.

Fidelity guard: neighborhood matrices are rebuilt with TopoBench's own
``select_neighborhoods_of_interest`` from masked ``incidence_r``
constituents, and at construction the full-coalition rebuild must
reproduce the batch's stored matrices exactly — otherwise construction
raises. TopoTune mutates ``x_{rank}`` in place, so every evaluation
snapshots and restores both features and neighborhood attributes.
"""

from __future__ import annotations

import torch

from topobench.data.utils.utils import select_neighborhoods_of_interest

from .explain import CellPlayer

__all__ = ["CellParticipationGame", "HopseParticipationGame",
           "pool_faithful_margin", "pool_faithful_logit",
           "hopse_pool_faithful_margin"]


def pool_faithful_margin(model, pos_cls: int, neg_cls: int,
                         encode: bool = False):
    """Build a ``model_fn`` whose readout pooling excludes detached cells.

    Detached rank-0 cells acquire nonzero "ghost" rows in the final
    ``x_0`` whenever a biased stage writes to rank 0 (a backbone route
    targeting rank 0, or PropagateSignalDown's LayerNorm/projector
    biases). Sum-pooling those rows credits absent cells and breaks the
    induced-subcomplex semantics (measured leak on MANTRA: up to 2.6
    margin units for the anchored model). Zeroing the absent rows after
    the readout and before pooling is exactly equivalent to physically
    dropping them under sum pooling.

    The participation games publish the detached indices on the batch
    as ``toposhap_absent`` ({rank: sorted indices}); this reads rank 0
    back (base readouts pool rank 0 only).
    """
    if model.readout.pooling_type != "sum":
        raise NotImplementedError(
            "pool-faithful margin is exact only for sum pooling; got "
            f"{model.readout.pooling_type!r}")

    def model_fn(b):
        if encode:
            b = model.feature_encoder(b)
        o = model.backbone(b)
        o = model.readout.forward(model_out=o, batch=b)
        x0 = o["x_0"]
        absent = getattr(b, "toposhap_absent", None) or {}
        if absent.get(0):
            x0 = x0.clone()
            x0[list(absent[0])] = 0.0
        logits = model.readout.compute_logits(
            x0, getattr(b, "batch_0", None))
        return float(logits[0, pos_cls] - logits[0, neg_cls])

    return model_fn


def pool_faithful_logit(model, target_cls: int, encode: bool = False):
    """Like :func:`pool_faithful_margin` but returns the TARGET-CLASS
    logit — the leg-1 GEA/GEF value convention (target = label class).
    """
    if model.readout.pooling_type != "sum":
        raise NotImplementedError(
            "pool-faithful logit is exact only for sum pooling; got "
            f"{model.readout.pooling_type!r}")

    def model_fn(b):
        if encode:
            b = model.feature_encoder(b)
        o = model.backbone(b)
        o = model.readout.forward(model_out=o, batch=b)
        x0 = o["x_0"]
        absent = getattr(b, "toposhap_absent", None) or {}
        if absent.get(0):
            x0 = x0.clone()
            x0[list(absent[0])] = 0.0
        logits = model.readout.compute_logits(
            x0, getattr(b, "batch_0", None))
        return float(logits[0, target_cls])

    return model_fn


def hopse_pool_faithful_margin(model, pos_cls: int, neg_cls: int):
    """Pool-faithful ``model_fn`` for HOPSE models (encoder included).

    ``HOPSEReadout`` pools EVERY rank (per-rank MLP -> scatter over
    ``batch_i`` -> concat -> MLP head), and the pooling happens inside
    its ``forward`` — so detached cells' ghost rows must be zeroed
    per rank, just before each scatter. When the batch carries no
    ``toposhap_absent`` marker (the game's pristine stored-encodings
    call), the stock readout runs instead, so the full-coalition guard
    checks this reimplementation against the stock path.
    """
    ro = model.readout
    if ro.task_level != "graph":
        raise NotImplementedError("graph-level HOPSE readout only")
    if ro.pooling_type != "sum":
        raise NotImplementedError(
            "pool-faithful margin is exact only for sum pooling; got "
            f"{ro.pooling_type!r}")

    def model_fn(b):
        b = model.feature_encoder(b)
        o = model.backbone(b)
        absent = getattr(b, "toposhap_absent", None)
        if absent is None:
            o = ro(model_out=o, batch=b)
            logits = o["logits"]
            return float(logits[0, pos_cls] - logits[0, neg_cls])
        from torch_scatter import scatter
        x_out = []
        for i in range(ro.complex_dim + 1):
            x_i = torch.cat(
                [o[f"x{i}_{j}"] for j in range(ro.max_hop)], dim=1)
            x_i = getattr(ro, f"linear_rank_{i}")(x_i)
            if absent.get(i):
                x_i = x_i.clone()
                x_i[list(absent[i])] = 0.0
            x_i = scatter(x_i, b[f"batch_{i}"], dim=0,
                          reduce=ro.pooling_type,
                          dim_size=int(b.batch_0.max()) + 1)
            x_out.append(x_i)
        logits = ro.linear(torch.cat(x_out, dim=1))
        return float(logits[0, pos_cls] - logits[0, neg_cls])

    return model_fn


def _mask_sparse(mat: torch.Tensor, absent_rows, absent_cols) -> torch.Tensor:
    """Drop entries whose row is in ``absent_rows`` or col in ``absent_cols``."""
    m = mat.coalesce()
    idx, val = m.indices(), m.values()
    keep = torch.ones(val.shape[0], dtype=torch.bool, device=val.device)
    if absent_rows:
        rows = torch.tensor(sorted(absent_rows), device=val.device)
        keep &= ~torch.isin(idx[0], rows)
    if absent_cols:
        cols = torch.tensor(sorted(absent_cols), device=val.device)
        keep &= ~torch.isin(idx[1], cols)
    return torch.sparse_coo_tensor(
        idx[:, keep], val[keep], m.size(), device=val.device
    ).coalesce()


def _abs_spmm(inc: torch.Tensor, x: torch.Tensor,
              transpose: bool = False) -> torch.Tensor:
    """|inc| @ x (or |inc|^T @ x) — ProjectionSum's projection step."""
    mat = torch.sparse_coo_tensor(
        inc.indices(), inc.values().abs(), inc.size()).coalesce()
    mat = mat.to(x.device)
    if transpose:
        mat = mat.t()
    return torch.sparse.mm(mat, x)


def _zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """Zero diagonal VALUES, keeping the entries in the pattern."""
    m = mat.coalesce()
    idx, val = m.indices(), m.values().clone()
    val[idx[0] == idx[1]] = 0.0
    return torch.sparse_coo_tensor(idx, val, m.size()).coalesce()


def _union_zero_diag(mat: torch.Tensor, present) -> torch.Tensor:
    """Ensure an explicit zero-valued diagonal entry for every PRESENT
    cell, matching the lift convention (the stored square matrices carry
    the full diagonal in the pattern even when the boundary product is
    empty — e.g. molecule complexes with no 2-cells). Absent cells get
    no entry, so detachment stays exact.
    """
    m = mat.coalesce()
    idx, val = m.indices(), m.values()
    have = set((idx[0][idx[0] == idx[1]]).tolist())
    add = [i for i in present if i not in have]
    if not add:
        return m
    di = torch.tensor(add, dtype=idx.dtype, device=idx.device)
    idx2 = torch.cat([idx, torch.stack([di, di])], dim=1)
    val2 = torch.cat([val, torch.zeros(len(add), dtype=val.dtype,
                                       device=val.device)])
    return torch.sparse_coo_tensor(idx2, val2, m.size()).coalesce()


def _base_connectivity(inc1: torch.Tensor, inc2: torch.Tensor,
                       present0=None, present1=None,
                       present2=None) -> dict:
    """The connectivity dict TopoBench's selector composes from.

    ``incidence_1`` is vertices x edges, ``incidence_2`` edges x
    triangles (signed, as stored on the batch). The lift stores 1-hop
    adjacency as the signed boundary product with the DIAGONAL VALUES
    ZEROED but the diagonal entries kept in the pattern — including
    diagonal entries the product never creates (full explicit diagonal
    over the rank's cells) — and sign-cancelled zero entries kept as
    explicit pattern members. TopoTune consumes both the index pattern
    (edge_index) and the values (edge_weight/edge_attr).
    """
    n0, n1 = inc1.size(0), inc1.size(1)
    n2 = inc2.size(1)
    p0 = range(n0) if present0 is None else sorted(present0)
    p1 = range(n1) if present1 is None else sorted(present1)
    p2 = range(n2) if present2 is None else sorted(present2)
    return {
        "incidence_1": inc1,
        "incidence_2": inc2,
        "adjacency_0": _union_zero_diag(
            _zero_diag(torch.sparse.mm(inc1, inc1.T)), p0),
        "adjacency_1": _union_zero_diag(
            _zero_diag(torch.sparse.mm(inc2, inc2.T)), p1),
        "coadjacency_1": _union_zero_diag(
            _zero_diag(torch.sparse.mm(inc1.T, inc1)), p1),
        "coadjacency_2": _union_zero_diag(
            _zero_diag(torch.sparse.mm(inc2.T, inc2)), p2),
    }


def _same_sparse(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact sparse equality; NaN entries count as equal to NaN.

    The lift's own 2-hop matrices carry NaN values (0/0 from its
    unit-ization of sign-cancelled entries), so faithful reproduction
    must place NaN where the stored matrix has NaN.
    """
    a, b = a.coalesce(), b.coalesce()
    if a.size() != b.size() or a.indices().shape != b.indices().shape:
        return False
    if not torch.equal(a.indices(), b.indices()):
        return False
    va, vb = a.values(), b.values()
    both_nan = torch.isnan(va) & torch.isnan(vb)
    return bool(torch.all((va == vb) | both_nan))


class CellParticipationGame:
    """v(S) = model output with only the cells in S participating.

    Parameters
    ----------
    model_fn : callable
        Post-encoder stages (backbone + readout) returning the scalar
        being explained — the same convention as ``CellMaskingGame``.
    batch : Data
        A single-complex batch carrying ``x_{rank}``, ``incidence_1``,
        ``incidence_2``, and the model's neighborhood attributes.
    players : list[CellPlayer]
        Cells of any rank; bit order = player order.
    encoder : callable, optional
        Applied once at construction (cellwise feature encoders commute
        with per-cell masking, so encode-once is exact and cheap).
    strict : bool
        Raise when the full-coalition rebuild does not reproduce the
        stored neighborhood matrices (default True).
    """

    def __init__(self, model_fn, batch, players: list[CellPlayer],
                 encoder=None, strict: bool = True,
                 lift_consistent: bool = False):
        self.model_fn = model_fn
        self.lift_consistent = lift_consistent
        self._encoder = encoder if lift_consistent else None
        if encoder is not None and not lift_consistent:
            with torch.no_grad():
                batch = encoder(batch)
        self.batch = batch
        self.players = players

        self._x0 = {}
        rank = 0
        while hasattr(batch, f"x_{rank}"):
            self._x0[rank] = getattr(batch, f"x_{rank}").clone()
            rank += 1

        # Lift-consistent mode: "absent" extends to cached aggregates.
        # ProjectionSum precomputes x_r = |B_r|^T x_{r-1} at preprocessing
        # from the intact complex, so a higher cell's features remember
        # detached boundary cells (the feature-cache leak). Here the
        # chain is recomputed from the masked incidences per coalition,
        # for exactly the ranks that were lifted (detected below), and
        # the encoder is applied per evaluation on the recomputed raws.
        self._lifted_ranks: list[int] = []
        if lift_consistent:
            incs = {1: batch.incidence_1, 2: batch.incidence_2}
            for r in (1, 2):
                if r not in self._x0 or self._x0[r].shape[0] == 0:
                    continue
                proj = _abs_spmm(incs[r].coalesce(),
                                 self._x0[r - 1], transpose=True)
                if torch.allclose(proj, self._x0[r], atol=1e-4):
                    self._lifted_ranks.append(r)
            if strict and 2 in self._x0 and self._x0[2].shape[0] > 0 \
                    and 2 not in self._lifted_ranks:
                raise ValueError(
                    "lift_consistent=True but stored x_2 does not match "
                    "the ProjectionSum of x_1 — unknown feature-lift "
                    "convention; refusing to run an unfaithful game")

        self._inc1 = batch.incidence_1.coalesce().cpu()
        self._inc2 = batch.incidence_2.coalesce().cpu()
        self._device = self._x0[0].device

        # batch attributes named by neighborhood strings
        from toposhap.vocabulary import NEIGHBORHOODS
        self._nbhd_names = [n for n in NEIGHBORHOODS if hasattr(batch, n)]
        self._nbhd0 = {n: getattr(batch, n) for n in self._nbhd_names}

        self._nbhd0["incidence_1"] = batch.incidence_1
        self._nbhd0["incidence_2"] = batch.incidence_2
        if strict and self._nbhd_names:
            rebuilt = self._rebuild(frozenset())
            for n in self._nbhd_names:
                if not _same_sparse(rebuilt[n].cpu(),
                                    self._nbhd0[n].coalesce().cpu()):
                    raise ValueError(
                        f"full-coalition rebuild of {n!r} does not match "
                        "the stored matrix — lift convention mismatch; "
                        "refusing to run an unfaithful game"
                    )

    def _absent(self, mask: int) -> dict[int, set]:
        absent: dict[int, set] = {0: set(), 1: set(), 2: set()}
        for pos, p in enumerate(self.players):
            if not (mask >> pos & 1):
                absent.setdefault(p.rank, set()).add(p.index)
        return absent

    def _rebuild(self, absent_key) -> dict:
        absent = dict(absent_key) if absent_key else {}
        a0 = set(absent.get(0, ()))
        a1 = set(absent.get(1, ()))
        a2 = set(absent.get(2, ()))
        inc1 = _mask_sparse(self._inc1, a0, a1)
        inc2 = _mask_sparse(self._inc2, a1, a2)
        conn = _base_connectivity(
            inc1, inc2,
            present0=[i for i in range(inc1.size(0)) if i not in a0],
            present1=[i for i in range(inc1.size(1)) if i not in a1],
            present2=[i for i in range(inc2.size(1)) if i not in a2])
        out = select_neighborhoods_of_interest(conn, self._nbhd_names)
        out = {n: m.coalesce().to(self._device) for n, m in out.items()}
        # readouts (e.g. PropagateSignalDown) read raw incidences directly
        out["incidence_1"] = inc1.to(self._device)
        out["incidence_2"] = inc2.to(self._device)
        return out

    def __call__(self, mask: int) -> float:
        absent = self._absent(mask)
        key = tuple((r, tuple(sorted(v))) for r, v in absent.items() if v)
        try:
            rebuilt = self._rebuild(key)
            for n, m in rebuilt.items():
                setattr(self.batch, n, m)
            xs = {}
            for rank, original in self._x0.items():
                x = original.clone()
                for i in absent.get(rank, ()):  # detach features + pooling
                    x[i] = 0.0
                xs[rank] = x
            for rank in self._lifted_ranks:
                # recompute the ProjectionSum cache over PRESENT cells;
                # masked incidence columns of absent cells are zero, so
                # their rows land at zero as required
                xs[rank] = _abs_spmm(
                    rebuilt[f"incidence_{rank}"].coalesce(),
                    xs[rank - 1], transpose=True)
            for rank, x in xs.items():
                setattr(self.batch, f"x_{rank}", x)
            # pool-faithful model_fns read this to drop detached cells
            # from readout pooling (see pool_faithful_margin)
            self.batch.toposhap_absent = {
                r: sorted(v) for r, v in absent.items()}
            with torch.no_grad():
                if self._encoder is not None:
                    self._encoder(self.batch)
                return float(self.model_fn(self.batch))
        finally:
            if hasattr(self.batch, "toposhap_absent"):
                del self.batch.toposhap_absent
            for n, m in self._nbhd0.items():
                setattr(self.batch, n, m)
            for rank, original in self._x0.items():
                setattr(self.batch, f"x_{rank}", original)


class HopseParticipationGame:
    """Participation game for HOPSE models: encodings are recomputed
    from the detached complex at every evaluation.

    Operates on the UNCOLLATED dataset sample (the same substrate the
    preprocessing transform originally ran on) and collates per
    evaluation: per coalition the incidences are masked, the
    neighborhood matrices rebuilt, absent cells' feature rows zeroed,
    stored encodings dropped, the HOPSE transform re-run, absent rows
    zeroed in every produced tensor, and the collated batch fed to
    ``model_fn`` (which must run the FULL model — encoder included).

    Guard: the full-coalition recompute must reproduce the
    stored-encodings prediction within ``value_tol``.
    """

    def __init__(self, model_fn, sample, collate, players, transform,
                 hop_prefixes=("x0_", "x1_", "x2_"), value_tol=1e-3,
                 batch_extra=None):
        import copy as _copy

        self.model_fn = model_fn
        # Stored samples hold tensors that are VIEWS into a whole-dataset
        # storage; deepcopy would copy that full storage per evaluation
        # (~0.6 s/eval measured). Compact every tensor once so the
        # per-evaluation deepcopy touches only this complex's data.
        compact = sample.__class__()
        for k in sample.keys():
            v = sample[k]
            compact[k] = (v.detach().clone() if torch.is_tensor(v)
                          else _copy.deepcopy(v))
        self.sample = compact
        self.collate = collate
        self.players = players
        self.transform = transform
        self.hop_prefixes = hop_prefixes
        self.batch_extra = dict(batch_extra or {})
        self._copy = _copy

        self._inc1 = sample.incidence_1.coalesce().cpu()
        self._inc2 = sample.incidence_2.coalesce().cpu()
        from toposhap.vocabulary import NEIGHBORHOODS
        self._nbhd_names = [n for n in NEIGHBORHOODS if hasattr(sample, n)]

        with torch.no_grad():
            b = self.collate([_copy.deepcopy(sample)])
            for k, v in self.batch_extra.items():
                b[k] = v
            pristine = float(model_fn(b))
        full = (1 << len(players)) - 1
        v_full = self(full)
        if abs(v_full - pristine) > value_tol:
            raise ValueError(
                f"full-coalition recompute gives {v_full:.6f} but the "
                f"stored encodings give {pristine:.6f} — the transform "
                "re-run does not reproduce preprocessing; refusing"
            )

    def _absent(self, mask: int):
        absent = {0: set(), 1: set(), 2: set()}
        for pos, p in enumerate(self.players):
            if not (mask >> pos & 1):
                absent[p.rank].add(p.index)
        return absent

    def __call__(self, mask: int) -> float:
        absent = self._absent(mask)
        d = self._copy.deepcopy(self.sample)
        inc1 = _mask_sparse(self._inc1, absent[0], absent[1])
        inc2 = _mask_sparse(self._inc2, absent[1], absent[2])
        conn = _base_connectivity(
            inc1, inc2,
            present0=[i for i in range(inc1.size(0))
                      if i not in absent[0]],
            present1=[i for i in range(inc1.size(1))
                      if i not in absent[1]],
            present2=[i for i in range(inc2.size(1))
                      if i not in absent[2]])
        rebuilt = select_neighborhoods_of_interest(conn, self._nbhd_names)
        for n, m in rebuilt.items():
            setattr(d, n, m.coalesce())
        d.incidence_1, d.incidence_2 = inc1, inc2
        # graph-level encoders (HKdiag/RWSE/...) read edge_index, which
        # the stored post-transform samples no longer carry — derive the
        # 1-skeleton from the MASKED incidence so detachment reaches it
        idx = inc1.coalesce().indices()
        pairs = {}
        for v, e in zip(idx[0].tolist(), idx[1].tolist()):
            pairs.setdefault(e, []).append(v)
        und = [(a, b) for vs in pairs.values() if len(vs) == 2
               for a, b in ((vs[0], vs[1]), (vs[1], vs[0]))]
        d.edge_index = (torch.tensor(und, dtype=torch.long).t().contiguous()
                        if und else torch.zeros(2, 0, dtype=torch.long))
        d.num_nodes = int(getattr(d, "x_0").shape[0])
        for rank in (0, 1, 2):
            if hasattr(d, f"x_{rank}"):
                x = getattr(d, f"x_{rank}").clone()
                for i in absent[rank]:
                    x[i] = 0.0
                setattr(d, f"x_{rank}", x)
        for key in [k for k in list(d.keys())
                    if any(str(k).startswith(p) for p in self.hop_prefixes)]:
            del d[key]
        with torch.no_grad():
            d = self.transform.forward(d)
            for rank in (0, 1, 2):
                for key in ([f"x_{rank}"]
                            + [f"x{rank}_{h}" for h in range(6)]):
                    if hasattr(d, key):
                        x = getattr(d, key)
                        for i in absent[rank]:
                            if i < x.shape[0]:
                                x[i] = 0.0
            b = self.collate([d])
            for k, v in self.batch_extra.items():
                b[k] = v
            # pool-faithful model_fns read this to drop detached cells
            # from readout pooling (see hopse_pool_faithful_margin)
            b.toposhap_absent = {r: sorted(v) for r, v in absent.items()}
            return float(self.model_fn(b))
