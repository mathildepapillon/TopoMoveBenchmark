"""Measured-FLOPs profiling shared by the local marker arms.

Conventions mirror the frozen records exactly:

* GCCN / model passes (#14 flops_<ds>.json 'counter'):
  ``torch.utils.flop_counter.FlopCounterMode``; dense matmul/conv ops,
  sparse scatter propagation uncounted (identical accounting across arms);
  backward profiled through a scalar sum of logits.
* HOPSE (#13 flops13.json 'conventions'): dense via FlopCounterMode actual
  dispatch + actual backward; encoding PREPROCESSING priced with analytic
  interceptors (eigh 9n^3, SVD/pinv 14mn^2+8n^3, sparse-mm 2x exact
  multiplication count) — those preprocessing FLOPs are already measured in
  the frozen #13 record (results/preprocess/pre_<ds>_union.json) for the
  identical full-8-block transform and are cited, not re-estimated.
  Elementwise, reductions, softmax, norms, scatter/gather and the optimizer
  step are uncounted everywhere.

Profiling runs AFTER all training/evaluation of a cell so the extra
forward/backward passes cannot perturb the seeded training RNG streams.
"""

from __future__ import annotations

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

GCCN_COUNTER_NOTE = (
    "torch.utils.flop_counter.FlopCounterMode; dense matmul/conv ops, "
    "sparse scatter propagation uncounted (identical accounting across "
    "arms); backward profiled through a scalar sum of logits"
)

HOPSE_CONVENTIONS_NOTE = (
    "dense: torch.utils.flop_counter.FlopCounterMode, actual dispatch, "
    "actual backward; encoding preprocessing priced by #13's interceptors "
    "(eigh 9n^3; svd/pinv 14mn^2+8n^3; sparse_mm 2x exact multiplication "
    "count; intercepted backward 2x forward) and cited from the frozen "
    "measured record data/frozen/hopse13/results/preprocess/"
    "pre_<dataset>_union.json (identical transform, identical data); "
    "elementwise/reductions/softmax/norms/scatter-gather/optimizer "
    "uncounted"
)


def profile_train_epoch(model, loader, device) -> dict:
    """One training epoch under FlopCounterMode (forward + backward).

    The backward is driven through a scalar sum of logits (#14 convention);
    the optimizer step is uncounted (never run).
    """
    model.train()
    model.state_str = "Training"
    n_items = 0
    t0 = time.perf_counter()
    with FlopCounterMode(display=False) as counter:
        for batch in loader:
            batch = batch.to(device)
            batch["model_state"] = "Training"
            out = model.forward(batch)
            out["logits"].sum().backward()
            model.zero_grad(set_to_none=True)
            n_items += int(out["logits"].shape[0])
    return {
        "flops": int(counter.get_total_flops()),
        "seconds": time.perf_counter() - t0,
        "n_items": n_items,
    }


def profile_eval_pass(model, loader, device, state: str = "Validation") -> dict:
    """One forward-only pass over a dataloader under FlopCounterMode."""
    model.eval()
    model.state_str = state
    n_items = 0
    t0 = time.perf_counter()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        for batch in loader:
            batch = batch.to(device)
            batch["model_state"] = state
            out = model.forward(batch)
            n_items += int(out["logits"].shape[0])
    return {
        "flops": int(counter.get_total_flops()),
        "seconds": time.perf_counter() - t0,
        "n_items": n_items,
    }
