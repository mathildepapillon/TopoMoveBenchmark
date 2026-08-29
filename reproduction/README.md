# TopoSHAP

Cell-native, **exact Shapley-interaction explanations** for topological deep
learning models (CCNNs/GCCNs as implemented in
[TopoBench](https://github.com/geometric-intelligence/TopoBench)) — the first
post-hoc attribution method for topological message-passing networks.

**The primary open-source artifact is the TopoBench fork**: branch
`toposhap-upstream` in `external/TopoBench` vendors the method as
`topobench.explain` (plus datasets, selection pipeline, tutorial, tests) so
TopoBench users adopt it with no extra dependency — see
[UPSTREAM.md](UPSTREAM.md). This repository is the research companion: the
frozen experimental record, campaign runners, and the pipeline that
regenerates every number and figure in the paper (MIT licensed).

One framework, two instantiations:

1. **Explaining predictions** (`toposhap.cells`): which cells — nodes, edges,
   faces — drove this prediction, with axiomatic guarantees. Benchmarks:
   GraphXAI (Benzene, Fluoride-Carbonyl, Mutagenicity) + MANTRA.
2. **Explaining model performance → cheap architecture selection**
   (`toposhap.neighborhoods`, `toposhap.selection`): the 11 candidate
   neighborhoods of a GCCN as players in a cooperative game. One guided run
   (all-neighborhood stem → 128-pass sampled Shapley → prune to top-k →
   continue) replaces architecture sweeps at ~1.2–3.6 training-run
   equivalents.

Target venue: **ICLR 2027** (abstract Sep 11, paper Sep 16 AoE; results freeze
~Aug 22). Double-blind — "Anonymous authors" everywhere.

## Layout

```
src/toposhap/
  vocabulary.py      the 11-neighborhood vocabulary; bit order = player order
  shapley/           exact values + interactions; permutation-sampled estimator
  selection/         top-k / anchored / greedy; automatic-k rules; submodularity census
  neighborhoods/     coalition masking of TopoTune routes; the one-run recipe
  cells/             leg-1 cell-masking game and explanations
  metrics/           GEF/GEA; axioms as executable checks
  patches/           the 3 TopoBench bug fixes/guards (TOPOSHAP_INTERRANK_ORIENTATION)
  costs/             FLOPs counting (torch + eigh/spmm shim); unified cost table
  io/                frozen-result loaders; results-freeze manifest
external/TopoBench   pinned upstream @ 6d8953e7e170 (branch toposhap-pin)
experiments/         recipe / landscape runners, marker-arm verifier, N* recompute, sbatch
figures/             lead-figure pipeline (FLOPs axis, spec-enforcing)
paper/               ICLR LaTeX skeleton; every number is a macro from frozen data
scripts/             ingest_results.py / freeze_results.py / make_tables.py
data/                INGEST.md (old-cluster inventory) + frozen/ (results land here)
tests/               58 tests incl. TopoBench integration (all passing)
```

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-scatter torch-sparse torch-cluster --no-build-isolation \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# external/TopoBench is not shipped in this repo — clone the pin first
# (uv venvs have no setuptools; --no-build-isolation needs it):
git clone https://github.com/geometric-intelligence/TopoBench external/TopoBench
git -C external/TopoBench checkout 6d8953e7e170
uv pip install setuptools wheel
uv pip install -e external/TopoBench --no-build-isolation
pytest   # 58 tests on a fresh clone; +8 parity tests once the record is ingested
```

## Quickstart: explain a prediction

```python
import os, torch
os.environ["TOPOSHAP_INTERRANK_ORIENTATION"] = "fixed"  # patched semantics
from toposhap.patches import apply_all; apply_all()

from toposhap.cells import CellPlayer, explain_cells
from toposhap.neighborhoods import CoalitionMaskedBackbone
from toposhap.testing import two_triangle_batch, full_vocabulary_backbone

wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone())
batch = two_triangle_batch()

def model_fn(b):                     # the scalar being explained
    with torch.no_grad():
        return wrapped(b)[0].sum()   # readout over the node rank

players = [CellPlayer(rank=0, index=i) for i in range(4)] + [
    CellPlayer(rank=2, index=j) for j in range(2)
]
expl = explain_cells(model_fn, batch, players, interactions=True)
print(expl.phi)            # one attribution per cell, summing to the score
print(expl.interactions)   # pairwise Grabisch-Roubens interaction indices
```

For a trained TopoBench model the pattern is identical with
`model.backbone` wrapped and `model_fn` returning the target-class
logit — see `tests/test_parity_original.py` for the exact leg-1
semantics (`encoder=model.feature_encoder`, rank-dependent baselines).

Masking caveats that matter in practice: models that consume per-hop
encoding tensors (HOPSE) need `HopseCellMaskingGame` — plain `x_{rank}`
masking never reaches their computation — and `baseline="complex_mean"`
degenerates to a no-op when per-rank features are constant (use
`baseline="zeros"`); see `src/toposhap/cells/explain.py`.

## Status

The original 14-experiment record is ingested (provenance:
[data/INGEST.md](data/INGEST.md)) and was frozen 2026-08-18
(`data/frozen/MANIFEST.json`). A gap-fill campaign (2026-08-28/29,
documented in [EXPERIMENTS.md](EXPERIMENTS.md) §6) then extended the
frozen frames in place: B=1 arms on the four added datasets, HOPSE
selection on the node-level citation datasets, post-hoc FLOPs for the
campaign's B=2 runs, the MANTRA GCCN sweep, and PROTEINS faithfulness.
The record was re-frozen 2026-08-29 after the campaign's last arms
(manifest sha256 `0d0fddde2379b856…`); `freeze_results.py --check`
verifies integrity. Raw per-run records live under `results/`.

Regenerate every derived artifact — tests, tables, paper macros, all ten
figures, deck-number consistency checks — from the frozen data with one
command:

```bash
scripts/reproduce.sh            # add --with-paper to also build the PDF
```

Integrity: `python scripts/freeze_results.py --check` verifies nothing under
`data/frozen/` has drifted from the manifest. Known original-cluster-only inputs are
documented in [figures/GAPS.md](figures/GAPS.md).

## For collaborators

- **Reproduce any paper number**: `scripts/reproduce.sh` regenerates
  tests, tables, macros, and every figure from `data/frozen/`;
  `--with-paper` also builds the PDF. Individual generators:
  `scripts/make_tables.py --performance-table`,
  `scripts/make_paper_tables.py`, `figures/render_lead_figures.py`.
- **Run a new experiment**: copy the runner pattern in
  `experiments/run_ladder_b2.py` (or `run_gccn_sweep.py` for sweeps).
  Always export `TOPOSHAP_INTERRANK_ORIENTATION=fixed`; GraphXAI
  datasets also need `TOPOSHAP_GRAPHXAI_DATA` pointing at
  `external/TopoBench/datasets/graphxai`. One JSONL per (dataset,
  seed) under `results/<arm>/`, then an `experiments/assemble_*.py`
  folds it into `data/frozen/`.
- **Binding rules** (protocol, framing, writing) live in
  [CLAUDE.md](CLAUDE.md) — they apply to humans too.
- **Overleaf upload**: `scripts/make_overleaf_export.py` emits
  `overleaf_export/`, a self-contained folder that compiles as-is.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full remaining-work map,
[CLAUDE.md](CLAUDE.md) for the binding framing/reporting rules, and
[UPSTREAM.md](UPSTREAM.md) for the TopoBench integration branch
(`toposhap-upstream` in `external/TopoBench`) and PR plan.
