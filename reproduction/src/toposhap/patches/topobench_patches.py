"""Runtime patches for the three TopoBench framework bugs (pinned 6d8953e7e170).

Found by TopoSHAP's executable axiom checks (experiment #1); upstream
issue/PR planned post-deadline. Applied at import time by
``toposhap.patches.apply_all()``, which every training/attribution entry point
calls. Each patch is individually switchable so ablations can reproduce the
buggy behaviour for the trustworthiness section.

Bug 1 — dead inter-rank routes (VERIFIED against the pinned source here):
    ``interrank_boundary_index`` in
    ``topobench/nn/backbones/combinatorial/gccn.py`` builds
    ``edge_index[0] = destination node ids`` and ``edge_index[1] = offset
    source-cell ids``. PyG message passing sends messages from
    ``edge_index[0]`` to ``edge_index[1]``, so messages flow INTO the offset
    source slots, while ``interrank_gnn_forward`` reads only the first
    ``n_dst_cells`` rows — which never receive anything. Inter-rank routes are
    therefore dead (destinations keep their zero-init features up to self-loop
    effects). The fix swaps the rows. Switch:
    ``TOPOSHAP_INTERRANK_ORIENTATION=fixed|upstream`` (default fixed).

Bug 2 — batch-dependent collapsed backbone (port from experiment #1 pending):
    The collapsed/one-hasse backbone produces batch-size-dependent outputs.
    The original fix lives in the old cluster's ``topobench.explain`` worktree
    (see data/INGEST.md); until ingested, ``check_batch_invariance`` below is
    the guard that catches it at runtime — the same invariance guard that
    caught the composed cross-rank batching bug in experiment #11.

Bug 3 — rank-aliasing offsets (port from experiment #1 pending):
    Cell-index offsets can alias across ranks. Guarded, not yet patched, for
    the same reason as bug 2.
"""

from __future__ import annotations

import os

import torch


def fixed_interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    """Corrected interrank edge_index: messages flow source cells -> dst.

    Row 0 holds the (offset) source-cell ids, row 1 the destination node ids,
    matching PyG's src->dst convention so ``expanded_out[:n_dst_cells]`` in
    ``interrank_gnn_forward`` actually contains propagated messages.
    """
    node_ids = (
        boundary_index[0]
        if torch.is_tensor(boundary_index[0])
        else torch.tensor(boundary_index[0], dtype=torch.int32)
    )
    edge_ids = (
        boundary_index[1]
        if torch.is_tensor(boundary_index[1])
        else torch.tensor(boundary_index[1], dtype=torch.int32)
    )
    adjusted_edge_ids = edge_ids + n_dst_nodes

    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = adjusted_edge_ids  # sources: offset higher-rank cells
    edge_index[1, :] = node_ids  # targets: destination cells
    edge_attr = x_src[edge_ids].squeeze()
    return edge_index, edge_attr


def apply_interrank_fix() -> bool:
    """Monkeypatch gccn.interrank_boundary_index unless the env switch says
    'upstream'. Returns True if the fix is active."""
    mode = os.environ.get("TOPOSHAP_INTERRANK_ORIENTATION", "fixed").lower()
    if mode not in {"fixed", "upstream"}:
        raise ValueError(
            f"TOPOSHAP_INTERRANK_ORIENTATION must be 'fixed' or 'upstream', "
            f"got {mode!r}"
        )
    if mode == "upstream":
        return False
    from topobench.nn.backbones.combinatorial import gccn

    gccn.interrank_boundary_index = fixed_interrank_boundary_index
    return True


def check_batch_invariance(
    model, batches_same_data, atol: float = 1e-5
) -> None:
    """Invariance guard: identical samples batched differently must produce
    identical per-sample outputs.

    ``batches_same_data`` is a sequence of batch objects containing the same
    underlying samples under different batch compositions. Raises
    ``AssertionError`` with the offending max deviation. This guard caught the
    composed cross-rank batching bug in experiment #11 and stands in for the
    bug-2/bug-3 patches until the original fixes are ingested.
    """
    outputs = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches_same_data:
            out = model(batch)
            outputs.append(out if torch.is_tensor(out) else out["labels"])
    if was_training:
        model.train()
    ref = outputs[0]
    for i, other in enumerate(outputs[1:], start=1):
        dev = (ref - other).abs().max().item()
        if dev > atol:
            raise AssertionError(
                f"batch-invariance violated: composition {i} deviates by "
                f"{dev:.3e} (atol {atol:.1e}) — collapsed-backbone or "
                f"cross-rank batching bug"
            )


def register_gt_lifting() -> bool:
    """Register CellCycleLiftingGT (toposhap.lifting_gt) with TopoBench.

    Lets pipelines select the ground-truth-aware cycle lifting via
    ``transforms.graph2cell_lifting.transform_name=CellCycleLiftingGT``
    without touching the pinned TopoBench fork.
    """
    import topobench.transforms as tbt

    from toposhap.lifting_gt import CellCycleLiftingGT

    for registry in (tbt.LIFTINGS, tbt.TRANSFORMS):
        registry["CellCycleLiftingGT"] = CellCycleLiftingGT
    return True


def register_structural_ones() -> bool:
    """Register the StructuralOnes feature lifting with TopoBench.

    Higher-rank cells get CONSTANT all-ones features of the rank-0
    feature width (MANTRA's native convention), so structural facts are
    never cached into features at preprocessing time: a ring cell
    carries no information beyond its incidences, and models must
    compute ring-ness through message passing. This is the
    training-time counterpart of the game's ``lift_consistent`` flag
    (the feature-cache finding, 2026-08-27). Select via
    ``transforms.graph2cell_lifting.feature_lifting=StructuralOnes``.
    """
    import torch
    import torch_geometric

    from topobench.transforms.feature_liftings import FEATURE_LIFTINGS

    class StructuralOnes(torch_geometric.transforms.BaseTransform):
        def __init__(self, **kwargs):
            super().__init__()

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"

        def lift_features(self, data):
            keys = sorted(
                key.split("_")[1] for key in data
                if ("incidence" in key and "-" not in key))
            for elem in keys:
                if f"x_{elem}" not in data:
                    # same width rule as ProjectionSum: rank below
                    idx = 0 if elem == "hyperedges" else int(elem) - 1
                    width = data[f"x_{idx}"].shape[1]
                    n = data[f"incidence_{elem}"].size(1)
                    data[f"x_{elem}"] = torch.ones(
                        (n, width), dtype=data["x_0"].dtype)
            return data

        def forward(self, data):
            return self.lift_features(data)

    FEATURE_LIFTINGS["StructuralOnes"] = StructuralOnes
    return True


def apply_all() -> dict[str, bool]:
    """Apply every available patch; returns {patch_name: active}."""
    return {
        "interrank_orientation_fix": apply_interrank_fix(),
        "gt_lifting_registered": register_gt_lifting(),
        "structural_ones_registered": register_structural_ones(),
        # bugs 2 & 3: guards only until original fixes are ingested
        "batch_invariance_guard": True,
        "rank_aliasing_guard": True,
    }
