"""The interrank orientation fix and the invariance guards.

The integration half (does the fix actually bring dead routes to life on a
real TopoTune backbone?) lives in test_masking.py; here we pin down the unit
semantics of the patch itself.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toposhap.patches import fixed_interrank_boundary_index  # noqa: E402


def toy_boundary():
    # 3 destination cells, 2 source cells; src 0 bounds dst {0,1}, src 1
    # bounds dst {1,2}
    dst_ids = torch.tensor([0, 1, 1, 2])
    src_ids = torch.tensor([0, 0, 1, 1])
    x_src = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    return x_src, (dst_ids, src_ids), 3


def upstream_interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    """Verbatim upstream behaviour (pinned 6d8953e7e170) for comparison."""
    node_ids, edge_ids = boundary_index
    adjusted = edge_ids + n_dst_nodes
    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted
    return edge_index, x_src[edge_ids].squeeze()


def test_fix_is_exactly_the_row_swap():
    x_src, boundary, n_dst = toy_boundary()
    up_ei, up_attr = upstream_interrank_boundary_index(x_src, boundary, n_dst)
    fx_ei, fx_attr = fixed_interrank_boundary_index(x_src, boundary, n_dst)
    assert torch.equal(fx_ei[0], up_ei[1])
    assert torch.equal(fx_ei[1], up_ei[0])
    assert torch.equal(fx_attr, up_attr)


def test_fixed_orientation_targets_destinations():
    x_src, boundary, n_dst = toy_boundary()
    ei, _ = fixed_interrank_boundary_index(x_src, boundary, n_dst)
    # In PyG convention messages flow ei[0] -> ei[1]: targets must be the
    # destination slots [0, n_dst), sources the offset slots [n_dst, ...).
    assert int(ei[1].max()) < n_dst
    assert int(ei[0].min()) >= n_dst


def test_env_switch(monkeypatch):
    pytest.importorskip("topobench")
    from topobench.nn.backbones.combinatorial import gccn

    from toposhap.patches import apply_interrank_fix

    original = gccn.interrank_boundary_index
    try:
        monkeypatch.setenv("TOPOSHAP_INTERRANK_ORIENTATION", "upstream")
        assert apply_interrank_fix() is False

        monkeypatch.setenv("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")
        assert apply_interrank_fix() is True
        assert gccn.interrank_boundary_index is fixed_interrank_boundary_index

        monkeypatch.setenv("TOPOSHAP_INTERRANK_ORIENTATION", "bogus")
        with pytest.raises(ValueError):
            apply_interrank_fix()
    finally:
        gccn.interrank_boundary_index = original


def test_batch_invariance_guard_fires():
    from toposhap.patches import check_batch_invariance

    class FakeModel(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, batch):
            return batch + self.offset

    model = FakeModel(0.0)
    same = torch.ones(4)
    check_batch_invariance(model, [same, same.clone()])  # must pass

    class BatchDependent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return batch + self.calls  # output depends on call order

    with pytest.raises(AssertionError, match="batch-invariance violated"):
        check_batch_invariance(BatchDependent(), [same, same.clone()])


def test_efficiency_and_null_player_checks():
    from toposhap.metrics import check_efficiency, find_null_players
    from toposhap.shapley import TabulatedGame, shapley_values

    n = 5
    rng = np.random.default_rng(0)
    # player 3 is null — the dead-route signature the axiom checks caught
    base = {m: float(rng.normal()) for m in range(1 << (n - 1))}

    def squeeze(mask):
        low = mask & 0b0111
        high = (mask >> 4) << 3
        return base[low | high]

    game = TabulatedGame(
        n_players=n, values={m: squeeze(m) for m in range(1 << n)}
    )
    phi = shapley_values(game, n)
    check_efficiency(phi, game, n)
    assert find_null_players(game, n) == [3]
