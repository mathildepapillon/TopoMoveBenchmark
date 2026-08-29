"""Submodularity census over a tabulated game.

The greedy selector's Nemhauser (1-1/e) guarantee holds iff the game is
submodular. We measure this rather than assume it: for every (S, T, i) with
S ⊂ T, i ∉ T, submodularity requires

    v(S+i) - v(S)  >=  v(T+i) - v(T).

Enumerating all such triples is O(3^n * n)-ish and infeasible even at n=11, so
the census follows the #14 protocol: NON-EMPTY triples only (S non-empty),
sampled exhaustively for small n or uniformly for large tables, reporting the
violation rate and the magnitude distribution of violations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposhap.shapley.games import TabulatedGame


@dataclass
class SubmodularityCensus:
    triples_checked: int
    violations: int
    violation_rate: float
    worst_violation: float  # most negative (v(S+i)-v(S)) - (v(T+i)-v(T))
    mean_violation: float  # mean magnitude among violations (0 if none)


def submodularity_census(
    game: TabulatedGame,
    max_triples: int = 2_000_000,
    seed: int = 0,
    atol: float = 1e-12,
) -> SubmodularityCensus:
    """Census of submodularity violations on non-empty (S ⊂ T, i ∉ T) triples.

    If the number of candidate triples exceeds ``max_triples``, samples
    uniformly with the given seed (reported counts are then estimates).
    ``atol`` separates violations from float noise; for measured (retrained)
    landscapes pass the dataset's retraining noise sd instead of the default.
    """
    n = game.n_players
    masks = sorted(game.values)
    rng = np.random.default_rng(seed)

    def random_triple():
        while True:
            t_mask = int(rng.integers(1, 1 << n))
            free = [i for i in range(n) if not t_mask >> i & 1]
            if not free:
                continue
            i = int(rng.choice(free))
            # random non-empty strict subset of t_mask
            bits = [b for b in range(n) if t_mask >> b & 1]
            if len(bits) < 2:
                continue
            keep = rng.random(len(bits)) < 0.5
            s_mask = sum(1 << b for b, k in zip(bits, keep) if k)
            if s_mask == 0 or s_mask == t_mask:
                continue
            return s_mask, t_mask, i

    def exhaustive_triples():
        for t_mask in masks:
            if t_mask == 0:
                continue
            free = [i for i in range(n) if not t_mask >> i & 1]
            sub = t_mask
            while True:
                sub = (sub - 1) & t_mask
                if sub == 0:
                    break
                for i in free:
                    yield sub, t_mask, i

    checked = violations = 0
    worst = 0.0
    viol_sum = 0.0

    if not game.is_exhaustive:
        raise ValueError("submodularity census needs an exhaustive game table")

    total_estimate = sum(
        (2 ** bin(t).count("1") - 2) * (n - bin(t).count("1")) for t in masks
    )
    use_sampling = total_estimate > max_triples

    source = (
        (random_triple() for _ in range(max_triples))
        if use_sampling
        else exhaustive_triples()
    )
    v = game.values
    for s_mask, t_mask, i in source:
        bit = 1 << i
        gap = (v[s_mask | bit] - v[s_mask]) - (v[t_mask | bit] - v[t_mask])
        checked += 1
        if gap < -atol:
            violations += 1
            viol_sum += -gap
            worst = min(worst, gap)

    return SubmodularityCensus(
        triples_checked=checked,
        violations=violations,
        violation_rate=violations / checked if checked else 0.0,
        worst_violation=worst,
        mean_violation=viol_sum / violations if violations else 0.0,
    )
