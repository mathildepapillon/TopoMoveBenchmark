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
