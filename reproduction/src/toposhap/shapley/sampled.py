"""Permutation-sampled Shapley estimator — the production attribution engine.

One "pass" is one uniformly random permutation of the players: walking the
permutation front to back yields one marginal contribution per player, so a
budget of P passes costs at most n*P + 1 distinct game evaluations (far fewer
in practice with caching, since prefixes repeat across permutations).

Production budgets from the experiment record: 128 passes for the one-run
recipe; 512 passes for stability resampling (cheap_sim12). Raising the budget
does NOT fix substitute-commitment picks — on Benzene seed 43 and Fluoride
seeds 42/43, 512-pass resamples became *more* confident in adjacency-dropping
picks (50/50 resamples). That is stem-model heterogeneity, not sampling noise;
only borderline seeds (e.g. 44) are budget-fixable. The estimator therefore
also reports per-player between-pass variance so downstream selectors can see
their own uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposhap.shapley.games import Game


@dataclass
class SampledAttribution:
    """Result of a permutation-sampled Shapley run."""

    phi: np.ndarray  # point estimate per player
    stderr: np.ndarray  # standard error of the mean, per player
    passes: int  # permutations consumed
    evaluations: int  # distinct game evaluations (honest cost)


def sampled_shapley(
    game: Game,
    n_players: int,
    passes: int = 128,
    seed: int | None = None,
    antithetic: bool = True,
) -> SampledAttribution:
    """Permutation-sampling estimate of Shapley values.

    With ``antithetic=True`` (default, matches production), each drawn
    permutation is paired with its reverse, which cancels a first-order term of
    the variance at no extra evaluation cost beyond the shared cache.
    """
    rng = np.random.default_rng(seed)
    n = n_players

    cache: dict[int, float] = {}
    evaluations = 0

    def v(mask: int) -> float:
        nonlocal evaluations
        if mask not in cache:
            cache[mask] = float(game(mask))
            evaluations += 1
        return cache[mask]

    marginals = np.zeros((passes, n))
    n_drawn = 0
    while n_drawn < passes:
        perm = rng.permutation(n)
        variants = [perm, perm[::-1]] if antithetic else [perm]
        for p in variants:
            if n_drawn >= passes:
                break
            mask = 0
            prev = v(0)
            for player in p:
                mask |= 1 << int(player)
                cur = v(mask)
                marginals[n_drawn, player] = cur - prev
                prev = cur
            n_drawn += 1

    phi = marginals.mean(axis=0)
    stderr = marginals.std(axis=0, ddof=1) / np.sqrt(passes)
    return SampledAttribution(
        phi=phi, stderr=stderr, passes=passes, evaluations=evaluations
    )


def resample_pick_stability(
    game: Game,
    n_players: int,
    passes: int,
    k: int,
    n_resamples: int = 50,
    seed: int = 0,
    selector=None,
    antithetic: bool = True,
) -> dict:
    """Fraction of independent re-estimates that agree on the selected subset.

    Mirrors the cheap_sim12 resampling protocol: run the sampled estimator
    ``n_resamples`` times at the given budget, apply the selector (default:
    plain top-k by Shapley value), and tabulate how often each mask is picked.
    A 50/50 split that persists as ``passes`` grows is the substitute-
    commitment signature.

    Caveat worth knowing: with ``antithetic=True``, EXACT substitutes receive
    identical estimates every run (each permutation's reverse hands the payoff
    to the other substitute), so deterministic tie-breaking masks their
    instability; real stems are never exactly symmetric, but synthetic probes
    should pass ``antithetic=False``.
    """
    from toposhap.selection.selectors import top_k_mask

    pick = selector or (lambda phi: top_k_mask(phi, k))
    counts: dict[int, int] = {}
    for r in range(n_resamples):
        att = sampled_shapley(
            game, n_players, passes=passes, seed=seed + r,
            antithetic=antithetic,
        )
        mask = pick(att.phi)
        counts[mask] = counts.get(mask, 0) + 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "modal_agreement": max(counts.values()) / total,
        "passes": passes,
        "n_resamples": n_resamples,
    }
