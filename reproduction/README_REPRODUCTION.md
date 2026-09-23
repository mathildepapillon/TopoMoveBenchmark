# Reproducing the TopoSHAP paper from this branch

This branch layers the paper's full experimental pipeline on top of the
`toposhap-explain` branch (the `topobench.explain` library). Everything
under `reproduction/` is the research companion: campaign runners,
assemblers, the frozen-record loaders, and the generators that produce
every table, macro, and figure in the paper.

## Layout

- repository root: TopoBench equipped with `topobench.explain`
  (tutorial in `tutorials/tutorial_explain.ipynb`).
- `reproduction/src/toposhap`: the research package (frozen
  11-neighborhood vocabulary, games, cost accounting, frozen IO).
- `reproduction/experiments/`: one runner + one assembler per arm.
- `reproduction/scripts/`: `reproduce.sh` (tests -> tables -> macros ->
  figures), table generators, freeze integrity.
- `reproduction/results/`: raw per-run JSONL records of the local
  campaign (checkpoints excluded).
- `reproduction/data/frozen/`: NOT in git (1.8 GB). Obtain per
  `reproduction/data/INGEST.md`, or from the authors' release tarball,
  then verify with `python scripts/freeze_results.py --check`.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install setuptools wheel
uv pip install -e .                       # this TopoBench, with explain
uv pip install -e "reproduction/[dev]"
uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-scatter torch-sparse torch-cluster --no-build-isolation \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## Reproduce

```bash
cd reproduction && scripts/reproduce.sh
```

## Provenance note

The recorded experiments ran against the pinned upstream commit
(branch `toposhap-pin`, 6d8953e7e170) with the runtime patches in
`reproduction/src/toposhap/patches` (applied by every runner). This
branch is based on `toposhap-explain`, which carries the interrank fix
natively; the runtime patch is a no-op there. For bit-exact reruns of
the historical record, check out `toposhap-pin` and install it in place
of the root package. `reproduction/EXPERIMENTS.md` maps every
arm to its runner and frozen frame; the exact selection protocol and
measurement conventions are in the paper's experimental appendix.
