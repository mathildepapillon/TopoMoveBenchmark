"""Cell masking for models that consume per-hop encodings (HOPSE).

HOPSE-style models assemble their input from the per-hop tensors
``x{rank}_{hop}`` and never read ``x_{rank}``, so the plain
:class:`CellMaskingGame` must be a structural no-op on them
(``v(full) == v(empty)`` exactly) while :class:`HopseCellMaskingGame`,
which masks the per-hop rows the model actually consumes, must not be.
"""

import pytest
import torch

from test.explain.fixtures import two_triangle_batch
from topobench.explain import (
    CellMaskingGame,
    CellPlayer,
    HopseCellMaskingGame,
    check_efficiency,
    shapley_values,
)
from topobench.explain.games import CachedGame

CHANNELS = 8
MAX_HOP = 2
N_CELLS = {0: 4, 1: 5, 2: 2}


def hopse_batch(seed=0):
    """Extend the two-triangle batch with raw per-hop tensors.

    The batch keeps its per-rank ``x_{rank}`` matrices (so the plain
    game can snapshot and mask them) and gains one raw ``x{rank}_{hop}``
    tensor per rank and hop, mimicking what the HOPSE transform provides
    before encoding.

    Parameters
    ----------
    seed : int
        Seed for the random per-rank and per-hop features.

    Returns
    -------
    types.SimpleNamespace
        Batch with x_{rank} and x{rank}_{hop} tensors.
    """
    batch = two_triangle_batch(channels=CHANNELS, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    for rank, n in N_CELLS.items():
        for hop in range(MAX_HOP):
            setattr(
                batch,
                f"x{rank}_{hop}",
                torch.randn(n, CHANNELS, generator=g),
            )
    return batch


class _HopEncoder(torch.nn.Module):
    """Per-(rank, hop) linear encoder writing ``x{rank}_{hop}`` in place.

    Mimics the HOPSE feature encoder's interface: it encodes every
    per-hop tensor and never touches ``x_{rank}``.
    """

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.linears = torch.nn.ModuleDict(
            {
                f"{rank}_{hop}": torch.nn.Linear(CHANNELS, CHANNELS)
                for rank in N_CELLS
                for hop in range(MAX_HOP)
            }
        )

    def forward(self, batch):
        for key, linear in self.linears.items():
            rank, hop = key.split("_")
            attr = f"x{rank}_{hop}"
            setattr(batch, attr, linear(getattr(batch, attr)))
        return batch


class _HopModel(torch.nn.Module):
    """HOPSE-style toy model: reads ONLY the per-hop tensors.

    Pools every ``x{rank}_{hop}`` over cells, mixes through a
    nonlinearity, and reads out one scalar. ``x_{rank}`` is never
    accessed, so masking it cannot reach the computation.
    """

    def __init__(self, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.mix = torch.nn.Linear(CHANNELS, CHANNELS)
        self.head = torch.nn.Linear(CHANNELS, 1)

    def forward(self, batch):
        pooled = sum(
            getattr(batch, f"x{rank}_{hop}").mean(dim=0)
            for rank in N_CELLS
            for hop in range(MAX_HOP)
        )
        return self.head(torch.relu(self.mix(pooled))).squeeze()


def make_game_parts(seed=3):
    """Build encoder, model, model_fn, and players for the hop games.

    Parameters
    ----------
    seed : int
        Base seed for the encoder and model weights.

    Returns
    -------
    tuple
        (encoder, model_fn, players).
    """
    encoder = _HopEncoder(seed=seed)
    encoder.eval()
    model = _HopModel(seed=seed + 1)
    model.eval()

    def model_fn(b):
        with torch.no_grad():
            return model(b)

    players = [CellPlayer(rank=0, index=i) for i in range(4)] + [
        CellPlayer(rank=2, index=j) for j in range(2)
    ]
    return encoder, model_fn, players


def test_plain_game_is_structural_noop_on_hop_consuming_model():
    """On a model reading only x{rank}_{hop}, masking x_{rank} is inert.

    The plain game must return the same value for the empty and the full
    coalition (bit-for-bit: nothing it masks is ever read), while the
    hop-masking game must separate them.
    """
    encoder, model_fn, players = make_game_parts()
    full = (1 << len(players)) - 1

    plain = CellMaskingGame(
        model_fn, hopse_batch(seed=3), players, encoder=encoder
    )
    assert plain(full) == plain(0)  # structural no-op, exact equality

    hop_game = HopseCellMaskingGame(
        model_fn,
        hopse_batch(seed=3),
        players,
        encoder=encoder,
        max_hop=MAX_HOP,
    )
    assert hop_game(full) != hop_game(0)

    # the grand coalition masks nothing: both games agree there
    assert abs(plain(full) - hop_game(full)) < 1e-6


def test_hopse_game_masks_every_hop_and_restores_the_batch():
    """v(empty) equals the model on zeroed player rows in every hop tensor,
    and the encoded originals survive evaluations."""
    encoder, model_fn, players = make_game_parts(seed=5)
    game = HopseCellMaskingGame(
        model_fn,
        hopse_batch(seed=5),
        players,
        encoder=encoder,
        max_hop=MAX_HOP,
    )

    with torch.no_grad():
        reference = encoder(hopse_batch(seed=5))
    for rank in (0, 2):
        for hop in range(MAX_HOP):
            x = getattr(reference, f"x{rank}_{hop}").clone()
            x[[p.index for p in players if p.rank == rank]] = 0.0
            setattr(reference, f"x{rank}_{hop}", x)
    with torch.no_grad():
        expected_empty = float(model_fn(reference))
    assert abs(game(0) - expected_empty) < 1e-6

    # snapshot/restore: the encoded originals survive evaluations
    for key, original in game._originals.items():
        assert torch.equal(getattr(game.batch, key), original)


def test_hopse_game_requires_encoded_hop_tensors():
    """Constructing with a hop the batch does not carry must fail loudly."""
    encoder, model_fn, players = make_game_parts(seed=7)
    with pytest.raises(ValueError, match="no encoded tensor"):
        HopseCellMaskingGame(
            model_fn,
            hopse_batch(seed=7),
            players,
            encoder=encoder,
            max_hop=MAX_HOP + 1,
        )


def test_hopse_game_efficiency():
    """Exact Shapley values on the hop-masking game satisfy efficiency."""
    encoder, model_fn, players = make_game_parts(seed=9)
    game = CachedGame(
        n_players=len(players),
        evaluate=HopseCellMaskingGame(
            model_fn,
            hopse_batch(seed=9),
            players,
            encoder=encoder,
            max_hop=MAX_HOP,
        ),
    )
    phi = shapley_values(game, len(players))
    check_efficiency(phi, game, len(players), atol=1e-4)


# ------------------------------------------------ the REAL HOPSE encoder

REAL_IN, REAL_OUT, REAL_N0 = 3, 4, 4


def real_hopse_parts(fuse_pse2cell=False, seed=11):
    """Smallest constructible setup around the real HOPSEFeatureEncoder.

    One dimension (rank 0), two hops, four nodes. The batch is a PyG
    ``Data`` object because the encoder's forward uses item access
    (``data["x0_0"]``) alongside attribute access (``data.batch_0``).

    Parameters
    ----------
    fuse_pse2cell : bool
        Whether the encoder also writes the fused ``x_0`` at encode time.
    seed : int
        Seed for the raw per-hop features and the model weights.

    Returns
    -------
    tuple
        (encoder, batch, model_fn, players); model_fn reads the per-hop
        tensors the way HOPSEWrapper assembles its backbone input.
    """
    from torch_geometric.data import Data

    from topobench.nn.encoders.hopse_encoder import HOPSEFeatureEncoder

    torch.manual_seed(seed)
    encoder = HOPSEFeatureEncoder(
        in_channels=[[REAL_IN] * MAX_HOP],
        out_channels=REAL_OUT,
        max_hop=MAX_HOP,
        fuse_pse2cell=fuse_pse2cell,
    )
    encoder.eval()

    g = torch.Generator().manual_seed(seed + 1)
    batch = Data(
        batch_0=torch.zeros(REAL_N0, dtype=torch.long),
        **{
            f"x0_{hop}": torch.randn(REAL_N0, REAL_IN, generator=g)
            for hop in range(MAX_HOP)
        },
    )

    head = torch.nn.Linear(REAL_OUT, 1)

    def model_fn(b):
        # what HOPSEWrapper does: assemble the input from x{rank}_{hop}
        pooled = sum(
            getattr(b, f"x0_{hop}").mean(dim=0) for hop in range(MAX_HOP)
        )
        return head(torch.tanh(pooled)).squeeze()

    players = [CellPlayer(rank=0, index=i) for i in range(REAL_N0)]
    return encoder, batch, model_fn, players


def test_hopse_game_against_the_real_encoder():
    """The game is live on the real HOPSEFeatureEncoder and satisfies
    efficiency with exact Shapley values."""
    encoder, batch, model_fn, players = real_hopse_parts()
    game = CachedGame(
        n_players=len(players),
        evaluate=HopseCellMaskingGame(
            model_fn, batch, players, encoder=encoder, max_hop=MAX_HOP
        ),
    )
    full = (1 << len(players)) - 1
    assert game(full) != game(0)
    phi = shapley_values(game, len(players))
    check_efficiency(phi, game, len(players), atol=1e-4)


def test_fused_encoder_is_refused_with_instructions():
    """fuse_pse2cell=True encoders must be refused at construction.

    In fused mode the encoder writes a fused ``x_0`` from the per-hop
    encodings at encode time, and fused-mode models (the graph HOPSE
    configs wrap a plain GNN reading ``batch.x_0``) consume that tensor.
    Post-encoding hop-masking never recomputes it, so the game would be
    a structural no-op; the error must name the working alternative.
    """
    encoder, batch, model_fn, players = real_hopse_parts(
        fuse_pse2cell=True
    )
    with pytest.raises(ValueError, match="fuse_pse2cell.*CellMaskingGame"):
        HopseCellMaskingGame(
            model_fn, batch, players, encoder=encoder, max_hop=MAX_HOP
        )


def test_fused_encoder_works_through_the_plain_game():
    """The refusal's instruction is honest: CellMaskingGame with the
    fused encoder masks the fused ``x_0`` rows a fused-mode model reads,
    and that game is live."""
    encoder, batch, _, players = real_hopse_parts(fuse_pse2cell=True)

    torch.manual_seed(13)
    head = torch.nn.Linear(REAL_OUT, 1)

    def fused_model_fn(b):
        # what GNNWrapper-style fused models do: read only x_0
        return head(torch.tanh(b.x_0.mean(dim=0))).squeeze()

    game = CellMaskingGame(
        fused_model_fn, batch, players, baseline="zeros", encoder=encoder
    )
    full = (1 << len(players)) - 1
    assert game(full) != game(0)
