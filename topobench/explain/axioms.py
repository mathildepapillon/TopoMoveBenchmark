"""Shapley axioms as executable checks.

Every attribution run can be audited against the axioms it claims, at a
cost that is small relative to the game evaluations it audits. These checks
have teeth: a player that the architecture says should matter but shows up
as a null player signals dead wiring, and the batch-invariance guard
catches models whose per-sample outputs depend on batch composition —
either failure silently corrupts any attribution computed on top.
"""

import numpy as np
import torch

from topobench.explain.games import Game


def check_efficiency(
    phi: np.ndarray, game: Game, n_players: int, atol: float = 1e-10
) -> float:
    """Assert the efficiency axiom: sum(phi) == v(N) - v(empty).

    Parameters
    ----------
    phi : np.ndarray
        Attribution per player.
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    atol : float
        Tolerance on the gap; for sampled attributions pass the
        estimator's tolerance instead of the exact default.

    Returns
    -------
    float
        The absolute efficiency gap (raises ``AssertionError`` if it
        exceeds ``atol``).
    """
    full = (1 << n_players) - 1
    gap = float(abs(phi.sum() - (game(full) - game(0))))
    if gap > atol:
        raise AssertionError(
            f"efficiency violated: gap {gap:.3e} > {atol:.1e}"
        )
    return gap


def find_null_players(
    game: Game,
    n_players: int,
    atol: float = 1e-10,
    n_probes: int | None = None,
) -> list[int]:
    """Find players whose marginal contribution is ~0 on every probe.

    Exhaustive when ``n_probes`` is None (2^(n-1) coalitions per player),
    otherwise probes that many random coalitions per player. A player that
    the architecture claims is live showing up here is a bug signal — e.g.
    a message-passing route whose contribution never reaches the output.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    atol : float
        Tolerance below which a marginal contribution counts as zero.
    n_probes : int, optional
        Number of random coalitions probed per player; None means
        exhaustive.

    Returns
    -------
    list[int]
        Bit indices of the null players found.
    """
    rng = np.random.default_rng(0)
    nulls = []
    for i in range(n_players):
        bit = 1 << i
        if n_probes is None:
            others = [m for m in range(1 << n_players) if not m & bit]
        else:
            others = [
                int(rng.integers(0, 1 << n_players)) & ~bit
                for _ in range(n_probes)
            ]
        if all(abs(game(m | bit) - game(m)) <= atol for m in others):
            nulls.append(i)
    return nulls


def check_symmetry_pair(game: Game, n_players: int, i: int, j: int) -> float:
    """Measure how far two players are from being symmetric in the game.

    Returns ``max |v(S+i) - v(S+j)|`` over all coalitions ``S`` excluding
    both, which is 0 iff ``i`` and ``j`` are symmetric. Symmetric players
    must receive equal Shapley values; symmetric players with unequal
    sampled attributions indicate estimator noise, not model structure.

    Parameters
    ----------
    game : Game
        Value function over coalition bitmasks.
    n_players : int
        Number of players.
    i : int
        Bit index of the first player.
    j : int
        Bit index of the second player.

    Returns
    -------
    float
        Worst-case asymmetry between the two players.
    """
    bi, bj = 1 << i, 1 << j
    worst = 0.0
    for m in range(1 << n_players):
        if m & (bi | bj):
            continue
        worst = max(worst, abs(game(m | bi) - game(m | bj)))
    return worst


def check_batch_invariance(
    model, batches_same_data, atol: float = 1e-5, key=None
) -> None:
    """Assert that batch composition does not change per-sample outputs.

    ``batches_same_data`` is a sequence of batch objects containing the
    same underlying samples under different batch compositions; a model
    whose per-sample outputs depend on composition silently corrupts any
    attribution computed on top of it. Raises ``AssertionError`` with the
    offending maximum deviation.

    Parameters
    ----------
    model : torch.nn.Module
        Model (or backbone) to evaluate.
    batches_same_data : Sequence
        Batch objects containing identical samples, batched differently.
    atol : float
        Maximum allowed deviation between compositions.
    key : str, optional
        When the model returns a mapping, the entry to compare; by default
        all entries are compared, flattened in sorted-key order.

    Returns
    -------
    None
        This check returns nothing; it raises on violation.
    """

    def flat(out):
        if torch.is_tensor(out):
            return out
        if key is not None:
            return out[key]
        return torch.cat([out[k].flatten() for k in sorted(out)])

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = [flat(model(batch)) for batch in batches_same_data]
    if was_training:
        model.train()
    ref = outputs[0]
    for i, other in enumerate(outputs[1:], start=1):
        dev = (ref - other).abs().max().item()
        if dev > atol:
            raise AssertionError(
                f"batch-invariance violated: composition {i} deviates by "
                f"{dev:.3e} (atol {atol:.1e})"
            )
