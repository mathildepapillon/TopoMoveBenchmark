"""Numerical parity against the ORIGINAL validated leg-1 explainer.

The original (experiment #1, exactness vs brute force to 2.254e-14) is
ingested read-only at
``data/frozen/reference/exp1_worktree/topobench_repo/topobench/explain/``.
These tests import it DIRECTLY from that path and check, on the two-triangle
fixture, that

* our ``CellMaskingGame`` in original mode (``encoder=...``, defaults) equals
  the original ``MaskedGCCNGame`` coalition-for-coalition — for the
  rank-dependent ``complex_mean`` baseline (our default with an encoder),
  ``zeros``, and explicit per-rank baseline rows (the ``train_mean``
  convention behind the frozen leg-1 numbers);
* both exact engines (the original's marginal-contribution oracle and Moebius
  path, our ``shapley_values``) agree at machine precision on the same game;
* the exact Shapley values of the two implementations agree end to end.

Import mechanics: the reference modules are loaded by file path. Their
``topobench.dataloader.utils.collate_fn`` import resolves against the
installed pinned TopoBench; ``brute_force.py``'s import of
``topobench.explain.interactions`` only exists in the reference repo copy, so
the reference ``interactions.py`` is registered under that dotted name first
(the installed ``topobench.explain`` — the upstream port of THIS repo — has
no ``interactions``/``masking``/``brute_force`` submodules, so nothing is
shadowed). See docs/parity-report.md for the full semantic diff.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("topobench")
pytest.importorskip("torch_geometric")

from toposhap.cells.explain import CellMaskingGame, CellPlayer  # noqa: E402
from toposhap.patches import apply_interrank_fix  # noqa: E402
from toposhap.shapley.exact import shapley_values  # noqa: E402
from toposhap.shapley.games import CachedGame, TabulatedGame  # noqa: E402
from toposhap.testing import (  # noqa: E402
    full_vocabulary_backbone,
    two_triangle_batch,
)
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REF_EXPLAIN = (
    REPO
    / "data"
    / "frozen"
    / "reference"
    / "exp1_worktree"
    / "topobench_repo"
    / "topobench"
    / "explain"
)


def _load_reference(dotted_name: str, filename: str):
    """Load a reference module from the frozen worktree by file path."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    path = REF_EXPLAIN / filename
    if not path.exists():
        # On a fresh clone the frozen reference worktree is absent until
        # ingestion; skip the parity gate rather than aborting collection
        # of the whole suite. CI on the ingested tree still runs it.
        pytest.skip(
            f"frozen reference module missing: {path} — parity gate needs "
            "the ingested record (see data/INGEST.md)",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


# Order matters: brute_force.py imports topobench.explain.interactions.
ref_interactions = _load_reference(
    "topobench.explain.interactions", "interactions.py"
)
ref_brute_force = _load_reference(
    "topobench.explain.brute_force", "brute_force.py"
)
ref_masking = _load_reference("topobench.explain.masking", "masking.py")

HIDDEN = 8
N_CLASSES = 3
RANKS = (0, 1, 2)
N_CELLS = {0: 4, 1: 5, 2: 2}
N_PLAYERS = sum(N_CELLS.values())

#: Cross-implementation tolerance (games share float64 math; observed ~1e-15).
GAME_TOL = 1e-6
#: Same-game engine-vs-engine tolerance (pure float64 algebra).
ENGINE_TOL = 1e-12


def two_triangle_data(channels: int = HIDDEN, seed: int = 0, label: int = 1):
    """The two-triangle fixture as a collate-able torch_geometric Data."""
    from torch_geometric.data import Data

    ns = two_triangle_batch(channels=channels, seed=seed)
    data = Data()
    for key in ("x_0", "x_1", "x_2", "cell_statistics"):
        data[key] = getattr(ns, key)
    for key in NEIGHBORHOODS:
        data[key] = getattr(ns, key)
    data["y"] = torch.tensor([label])
    return data


class _Encoder(torch.nn.Module):
    """Per-rank linear feature encoder (TopoBench-style: mutates + returns)."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.linears = torch.nn.ModuleDict(
            {str(r): torch.nn.Linear(channels, hidden) for r in RANKS}
        )

    def forward(self, batch):
        for r in RANKS:
            setattr(
                batch, f"x_{r}", self.linears[str(r)](getattr(batch, f"x_{r}"))
            )
        return batch


class _BackboneWrapper(torch.nn.Module):
    """Adapter: TopoTune's {rank: tensor} output -> TBModel model_out dict."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        out = self.backbone(batch)
        return {f"x_{r}": x for r, x in out.items()}


class _SumPoolReadout(torch.nn.Module):
    """Sum-pool rank-0 cells per complex, then one linear map to logits."""

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, n_classes)

    def forward(self, model_out, batch):
        x0 = model_out["x_0"]
        index = batch.batch_0.to(x0.device)
        pooled = torch.zeros(
            int(index.max()) + 1,
            x0.shape[1],
            dtype=x0.dtype,
            device=x0.device,
        )
        pooled.index_add_(0, index, x0)
        model_out["logits"] = self.linear(pooled)
        return model_out


class _Model(torch.nn.Module):
    """Minimal TBModel-shaped model: feature_encoder / backbone / readout."""

    def __init__(
        self, channels: int = HIDDEN, n_classes: int = N_CLASSES, seed: int = 0
    ):
        super().__init__()
        self.feature_encoder = _Encoder(channels, channels)
        self.backbone = _BackboneWrapper(
            full_vocabulary_backbone(channels=channels, seed=seed)
        )
        torch.manual_seed(seed + 1)
        self.readout = _SumPoolReadout(channels, n_classes)


def _random_coalitions(n_masks: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << N_PLAYERS, size=n_masks)


def _masks_to_bool(masks: np.ndarray) -> np.ndarray:
    """Bitmask ints -> boolean rows; bit b = player b in both codebases."""
    out = np.zeros((len(masks), N_PLAYERS), dtype=bool)
    for row, mask in enumerate(masks):
        for bit in range(N_PLAYERS):
            out[row, bit] = mask >> bit & 1
    return out


@pytest.fixture(scope="module")
def setup():
    """Shared model, data, original game and our game in original mode."""
    os.environ["TOPOSHAP_INTERRANK_ORIENTATION"] = "fixed"
    assert apply_interrank_fix() is True

    data = two_triangle_data()
    model = _Model()
    # The original game casts the (shared) model to float64/eval in place.
    orig = ref_masking.MaskedGCCNGame(
        model, data, baseline="complex_mean", target="label", batch_size=64
    )
    target = orig.target

    def model_fn(batch):
        model_out = model.backbone(batch)
        model_out = model.readout(model_out=model_out, batch=batch)
        return model_out["logits"][0, target]

    # Rank-major player order, matching the original's global indexing.
    players = [
        CellPlayer(rank=r, index=i) for r in RANKS for i in range(N_CELLS[r])
    ]

    def our_batch():
        return ref_masking.cast_batch(
            ref_masking.collate_single(data), torch.float64
        )

    ours = CellMaskingGame(
        model_fn, our_batch(), players, encoder=model.feature_encoder
    )
    return {
        "data": data,
        "model": model,
        "model_fn": model_fn,
        "players": players,
        "our_batch": our_batch,
        "orig": orig,
        "ours": ours,
        "target": target,
    }


def test_defaults_select_original_semantics(setup):
    """With an encoder, our defaults = the original's rank-dependent game."""
    assert setup["ours"].baseline_name == "complex_mean"
    for r in RANKS:
        dev = (
            (setup["ours"]._originals[r] - setup["orig"].h[r]).abs().max()
        )
        assert float(dev) == 0.0, f"encoded rank-{r} features differ"


def test_coalition_value_parity_complex_mean(setup):
    masks = _random_coalitions()
    v_orig = setup["orig"](_masks_to_bool(masks))
    v_ours = np.array([setup["ours"](int(m)) for m in masks])
    dev = float(np.abs(v_orig - v_ours).max())
    assert dev < GAME_TOL, f"complex_mean coalition values deviate by {dev:e}"


def test_coalition_value_parity_zeros(setup):
    orig = ref_masking.MaskedGCCNGame(
        setup["model"],
        setup["data"],
        baseline="zeros",
        target="label",
        batch_size=64,
    )
    ours = CellMaskingGame(
        setup["model_fn"],
        setup["our_batch"](),
        setup["players"],
        baseline="zeros",
        encoder=setup["model"].feature_encoder,
    )
    masks = _random_coalitions()
    v_orig = orig(_masks_to_bool(masks))
    v_ours = np.array([ours(int(m)) for m in masks])
    dev = float(np.abs(v_orig - v_ours).max())
    assert dev < GAME_TOL, f"zeros-baseline coalition values deviate by {dev:e}"


def test_coalition_value_parity_baseline_rows(setup):
    """Explicit per-rank rows: the original's train_mean convention."""
    g = torch.Generator().manual_seed(123)
    rows = {
        r: torch.randn(HIDDEN, generator=g, dtype=torch.float64) for r in RANKS
    }
    orig = ref_masking.MaskedGCCNGame(
        setup["model"],
        setup["data"],
        baseline=dict(rows),
        target="label",
        batch_size=64,
    )
    ours = CellMaskingGame(
        setup["model_fn"],
        setup["our_batch"](),
        setup["players"],
        baseline=dict(rows),
        encoder=setup["model"].feature_encoder,
    )
    masks = _random_coalitions()
    v_orig = orig(_masks_to_bool(masks))
    v_ours = np.array([ours(int(m)) for m in masks])
    dev = float(np.abs(v_orig - v_ours).max())
    assert dev < GAME_TOL, f"baseline-rows coalition values deviate by {dev:e}"


def test_full_value_is_target_class_logit(setup):
    """Value convention: v(full) = the model's target-class logit."""
    full = (1 << N_PLAYERS) - 1
    model_logit = float(setup["orig"].model_logits_full[setup["target"]])
    assert abs(setup["ours"](full) - model_logit) < 1e-9
    assert abs(setup["orig"].full_value - model_logit) < 1e-9


def test_legacy_raw_masking_is_a_different_game(setup):
    """The legacy default (raw rows, zeros) must stay a distinct option: it
    agrees with the original at the unmasked point and diverges under
    masking (encoded-vs-raw masking point genuinely differs)."""
    model, data = setup["model"], setup["data"]

    def raw_model_fn(batch):
        batch = model.feature_encoder(batch)
        model_out = model.backbone(batch)
        model_out = model.readout(model_out=model_out, batch=batch)
        return model_out["logits"][0, setup["target"]]

    raw = CellMaskingGame(
        raw_model_fn, setup["our_batch"](), setup["players"]
    )
    assert raw.baseline_name == "zeros"
    full = (1 << N_PLAYERS) - 1
    assert abs(raw(full) - setup["orig"].full_value) < 1e-9
    masks = _random_coalitions()
    v_orig = setup["orig"](_masks_to_bool(masks))
    v_raw = np.array([raw(int(m)) for m in masks])
    assert float(np.abs(v_orig - v_raw).max()) > 1e-6


def test_exact_engines_agree_on_the_same_game(setup):
    """Original oracle + original Moebius path + our engine, same values."""
    members = np.arange(N_PLAYERS)
    values = ref_brute_force.game_values_on_lattice(setup["orig"], members)
    centered = values - values[0]

    phi_oracle = ref_brute_force.brute_force_shapley(centered, N_PLAYERS)
    phi_moebius = ref_interactions.shapley_from_moebius(
        ref_interactions.subset_values_to_moebius(centered), N_PLAYERS
    )
    tab = TabulatedGame(
        n_players=N_PLAYERS,
        values={m: float(values[m]) for m in range(1 << N_PLAYERS)},
    )
    phi_ours = shapley_values(tab, N_PLAYERS)

    dev_om = float(np.abs(phi_oracle - phi_moebius).max())
    dev_ou = float(np.abs(phi_oracle - phi_ours).max())
    assert dev_om < ENGINE_TOL, f"original oracle vs Moebius: {dev_om:e}"
    assert dev_ou < ENGINE_TOL, f"original oracle vs our engine: {dev_ou:e}"


def test_exact_shapley_cross_implementation(setup):
    """End to end: our game in original mode + our engine vs the original."""
    members = np.arange(N_PLAYERS)
    values_orig = ref_brute_force.game_values_on_lattice(
        setup["orig"], members
    )
    game = CachedGame(n_players=N_PLAYERS, evaluate=setup["ours"])
    values_ours = np.array(
        [game(m) for m in range(1 << N_PLAYERS)], dtype=np.float64
    )
    lattice_dev = float(np.abs(values_orig - values_ours).max())
    assert lattice_dev < GAME_TOL, f"full lattice deviates by {lattice_dev:e}"

    phi_orig = ref_brute_force.brute_force_shapley(
        values_orig - values_orig[0], N_PLAYERS
    )
    phi_ours = shapley_values(game, N_PLAYERS)
    phi_dev = float(np.abs(phi_orig - phi_ours).max())
    assert phi_dev < GAME_TOL, f"exact Shapley values deviate by {phi_dev:e}"

    efficiency = float(
        abs(phi_ours.sum() - (values_ours[-1] - values_ours[0]))
    )
    assert efficiency < ENGINE_TOL
