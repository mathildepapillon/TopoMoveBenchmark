"""TopoBench bug patches and runtime guards."""

from toposhap.patches.topobench_patches import (
    apply_all,
    apply_interrank_fix,
    check_batch_invariance,
    fixed_interrank_boundary_index,
)

__all__ = [
    "apply_all",
    "apply_interrank_fix",
    "check_batch_invariance",
    "fixed_interrank_boundary_index",
]
