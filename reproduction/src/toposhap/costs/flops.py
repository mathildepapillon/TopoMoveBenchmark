"""FLOPs accounting: torch FlopCounterMode plus a shim for numpy/scipy ops.

The unified cost table (Stage D of experiment #13) prices every approach in
both wall-clock seconds and FLOPs. Torch ops are counted by
``torch.utils.flop_counter.FlopCounterMode``; preprocessing that happens in
numpy/scipy (Hodge spectral features via ``eigh``, sparse matmuls) is priced
from operand shapes by the shim below, mirroring the #13 methodology.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field


@dataclass
class FlopLedger:
    """Accumulates FLOPs from torch and shimmed non-torch sources."""

    torch_flops: int = 0
    shim_flops: int = 0
    shim_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.torch_flops + self.shim_flops

    def add_shim(self, op: str, flops: int) -> None:
        self.shim_flops += int(flops)
        self.shim_breakdown[op] = self.shim_breakdown.get(op, 0) + int(flops)

    # ---- shape-based prices for the non-torch ops used in this project ----

    def eigh(self, n: int) -> None:
        """Dense symmetric eigendecomposition of an n x n matrix: ~ (4/3) n^3
        for the reduction plus O(n^3) for eigenvectors; priced at 4 n^3 to be
        conservative (LAPACK syevd-style, matching #13's shim)."""
        self.add_shim("eigh", 4 * n**3)

    def spmm(self, nnz: int, dense_cols: int) -> None:
        """Sparse-dense matmul: 2 FLOPs per nonzero per output column."""
        self.add_shim("spmm", 2 * nnz * dense_cols)

    def matmul(self, m: int, k: int, n: int) -> None:
        """Dense matmul fallback for shim-side code paths."""
        self.add_shim("matmul", 2 * m * k * n)


@contextlib.contextmanager
def count_flops(ledger: FlopLedger):
    """Count torch FLOPs inside the block into ``ledger``.

    Non-torch ops must be priced explicitly through the ledger's shim methods
    at their call sites (there is no reliable way to hook numpy globally
    without perturbing timings).
    """
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        yield ledger
    ledger.torch_flops += counter.get_total_flops()
