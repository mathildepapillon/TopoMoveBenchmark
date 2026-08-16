"""Cooperative games over bitmask-encoded coalitions.

A game is any callable ``v(mask: int) -> float`` defined on coalitions of
``n`` players, where a coalition is encoded as an integer bitmask (bit ``i``
set means player ``i`` is present). Two concrete games are provided:

* :class:`TabulatedGame` — values precomputed for every subset, e.g. an
  exhaustive landscape of retrained models.
* :class:`CachedGame` — a memoizing wrapper around an expensive evaluator
  (e.g. a forward pass under coalition masking, or retrain-and-score) with
  an optional budget, so estimators can be priced by distinct evaluations.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

#: A cooperative game: coalition bitmask -> value.
Game = Callable[[int], float]


@dataclass
class TabulatedGame:
    """Game backed by a table of subset values.

    Missing masks raise ``KeyError``, so a partial table can never silently
    masquerade as an exhaustive one.

    Parameters
    ----------
    n_players : int
        Number of players in the game.
    values : Mapping[int, float]
        Mapping from coalition bitmask to game value.
    """

    n_players: int
    values: Mapping[int, float]

    def __call__(self, mask: int) -> float:
        """Evaluate the game on a coalition.

        Parameters
        ----------
        mask : int
            Coalition bitmask.

        Returns
        -------
        float
            The tabulated value of the coalition.
        """
        return float(self.values[mask])

    @property
    def is_exhaustive(self) -> bool:
        """Return whether every one of the 2^n coalitions is tabulated.

        Returns
        -------
        bool
            True if the table covers all coalitions.
        """
        return len(self.values) == 1 << self.n_players

    def argmax(self) -> tuple[int, float]:
        """Return the best coalition and its value over all tabulated subsets.

        Returns
        -------
        tuple[int, float]
            The best coalition bitmask and its game value.
        """
        best = max(self.values, key=lambda m: self.values[m])
        return best, float(self.values[best])

    def argmax_at_size(self, k: int) -> tuple[int, float]:
        """Return the best coalition of exactly k players.

        Parameters
        ----------
        k : int
            Coalition size to restrict the search to.

        Returns
        -------
        tuple[int, float]
            The best size-k coalition bitmask and its game value.
        """
        candidates = {
            m: v for m, v in self.values.items() if bin(m).count("1") == k
        }
        if not candidates:
            raise KeyError(f"no tabulated subsets of size {k}")
        best = max(candidates, key=lambda m: candidates[m])
        return best, float(candidates[best])


@dataclass
class CachedGame:
    """Memoizing wrapper around an expensive coalition evaluator.

    ``budget`` (if set) caps the number of *distinct* evaluations; exceeding
    it raises ``RuntimeError``. :attr:`calls` counts distinct evaluations for
    honest cost accounting — cache hits are free by construction and priced
    as such.

    Parameters
    ----------
    n_players : int
        Number of players in the game.
    evaluate : Callable[[int], float]
        Expensive evaluator mapping a coalition bitmask to its value.
    budget : int, optional
        Maximum number of distinct evaluations allowed.
    """

    n_players: int
    evaluate: Callable[[int], float]
    budget: int | None = None
    _cache: dict[int, float] = field(default_factory=dict, init=False)

    def __call__(self, mask: int) -> float:
        """Evaluate the game on a coalition, memoizing the result.

        Parameters
        ----------
        mask : int
            Coalition bitmask.

        Returns
        -------
        float
            The (possibly cached) value of the coalition.
        """
        if mask not in self._cache:
            if self.budget is not None and len(self._cache) >= self.budget:
                raise RuntimeError(
                    f"evaluation budget {self.budget} exhausted at mask {mask}"
                )
            self._cache[mask] = float(self.evaluate(mask))
        return self._cache[mask]

    @property
    def calls(self) -> int:
        """Return the number of distinct evaluations performed so far.

        Returns
        -------
        int
            Count of distinct coalition evaluations.
        """
        return len(self._cache)

    def table(self) -> dict[int, float]:
        """Return a snapshot of everything evaluated so far.

        Returns
        -------
        dict[int, float]
            Copy of the internal mask -> value cache.
        """
        return dict(self._cache)
