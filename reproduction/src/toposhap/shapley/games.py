"""Cooperative games over bitmask-encoded coalitions.

A game is any callable ``v(mask: int) -> float`` over coalitions of ``n``
players encoded as integer bitmasks (bit i = player i). Concrete games in this
codebase:

* :class:`TabulatedGame` — values precomputed for every subset (exhaustive
  landscapes, retraining games read from frozen results).
* :class:`CachedGame` — wraps an expensive evaluator (e.g. retrain-and-score,
  or a forward pass under coalition masking) with memoisation and a call
  budget, so estimators can be priced honestly.

Scoring convention: accuracy-scored games beat logit-scored games for
neighborhood attribution (experiment #8); model-facing game builders should
default to accuracy scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

Game = Callable[[int], float]


@dataclass
class TabulatedGame:
    """Game backed by a table of subset values.

    ``values`` maps bitmask -> value. Missing masks raise ``KeyError`` so a
    partial landscape can never silently masquerade as an exhaustive one.
    """

    n_players: int
    values: Mapping[int, float]

    def __call__(self, mask: int) -> float:
        return float(self.values[mask])

    @property
    def is_exhaustive(self) -> bool:
        return len(self.values) == 1 << self.n_players

    def argmax(self) -> tuple[int, float]:
        """Best coalition and its value over all tabulated subsets."""
        best = max(self.values, key=lambda m: self.values[m])
        return best, float(self.values[best])

    def argmax_at_size(self, k: int) -> tuple[int, float]:
        """Best coalition of exactly k players among tabulated subsets."""
        candidates = {
            m: v for m, v in self.values.items() if bin(m).count("1") == k
        }
        if not candidates:
            raise KeyError(f"no tabulated subsets of size {k}")
        best = max(candidates, key=lambda m: candidates[m])
        return best, float(candidates[best])


@dataclass
class CachedGame:
    """Memoising wrapper around an expensive coalition evaluator.

    ``budget`` (if set) caps the number of *distinct* evaluations; exceeding it
    raises ``RuntimeError``. ``calls`` counts distinct evaluations for honest
    cost accounting (cache hits are free by construction and priced as such).
    """

    n_players: int
    evaluate: Callable[[int], float]
    budget: int | None = None
    _cache: dict[int, float] = field(default_factory=dict)

    def __call__(self, mask: int) -> float:
        if mask not in self._cache:
            if self.budget is not None and len(self._cache) >= self.budget:
                raise RuntimeError(
                    f"evaluation budget {self.budget} exhausted at mask {mask}"
                )
            self._cache[mask] = float(self.evaluate(mask))
        return self._cache[mask]

    @property
    def calls(self) -> int:
        return len(self._cache)

    def table(self) -> dict[int, float]:
        """Snapshot of everything evaluated so far (for freezing to disk)."""
        return dict(self._cache)
