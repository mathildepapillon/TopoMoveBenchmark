# Remaining work → where it lives in this repo

Timeline: results freeze target ~Aug 22 · complete draft Aug 26 · ICLR
abstract Sep 11 · full paper Sep 16 (AoE).

## 0. Ingestion (blocks everything below)

The 14-experiment record lives on the the original compute cluster, which this machine
cannot reach (verified: no mount, no SSH route, no shared credentials). The
easiest path: run [scripts/make_ingest_bundle.sh](scripts/make_ingest_bundle.sh)
ON the original compute cluster (or paste it to the old thread's agent), move the single tarball
here, then:

```bash
mkdir -p data/_staging && tar -xzf toposhap_ingest_*.tar.gz -C data/_staging
python scripts/ingest_results.py --staging data/_staging
python scripts/ingest_results.py --verify
```

In-flight there at handoff: **#13** (`[record-id]…`, HOPSE recipe +
Stage D unified cost table) and **#14** (`[record-id]…`, 5-seed marker
arm with anchored selectors + automatic-k battery + submodularity census;
gate job 68888). If its interactive SSH lane is wedged again: the managed
submission lane works — escalate rather than letting it retry `salloc`.

## 1. The 8-hour ask: regenerated lead figure (was due ~Aug 17 UTC)

On #14 marker-arm arrival:

```bash
python experiments/verify_marker_arm.py          # sd collapse + mean hold per cell
python scripts/make_tables.py --menu-per-seed --ceilings
python figures/build_lead_figure.py --selector anchored --x flops
```

Acceptance: (a) anchored stars with tight bars (±0.004–0.014, training-noise
width); (b) no accuracy downgrade vs plain; (c) FLOPs x-axis. Predictions on
record: Benzene k=2/3 bars collapse ~25×; Fluoride k=2/3 means +0.05. Any
anchored cell below plain → shown honestly, treated as the measurable k=2
exception, flagged to the researcher immediately (the verifier exits 2 and
prints `!!` rows).

## 2. #13 review on landing

HOPSE verdicts under **both** attribution accountings; Stage D cost table
(wall-clock + FLOPs) → `data/frozen/cost_table13.parquet`; backfill the
`flops` column of `menu_per_seed_selection.parquet` (see make_tables.py).

## 3. Recompute N*/regrets from final data

```bash
python experiments/recompute_nstar.py --selector anchored
```

Fluoride/Cora misses are diagnosed (adjacency-drop); the anchored arm is
expected to repair them.

## 4. New runs on THIS cluster (if needed before freeze)

`experiments/run_recipe.py` + `experiments/slurm/recipe.sbatch` (single GPU)
and `experiments/slurm/landscape_array.sbatch` (array, `%48` throttle,
background priority — first cut if compute tightens; pre-registered fallback
is size≤4 exhaustive). Two deliberate `NotImplementedError` seams in
`TopoBenchHooks` must be wired on the first GPU run — they are review gates,
not omissions. Re-measure per-run seconds on H200s before pricing anything
(experiments/configs/datasets.yaml carries the old H100 numbers as hints).

## 5. Freeze → paper

```bash
python scripts/freeze_results.py --note "results freeze"
python scripts/make_tables.py --all      # tables + macros + manifest stamp
cd paper && make                          # or tectonic main.tex
```

Paper skeleton: TopoTune-style, OP-indexed contributions, boxed T1/T2 theory
block (statements TODO researcher), claim-paragraph experiment sections,
every number an `\unfrozen{\Res…}` macro until the freeze flips
`\FrozenResultstrue`. Cut from this submission: hypergraph experiment
(carried by corollary T2), B-XAIC, ShapeGGen.

## Parity checks before freeze (from the fresh-codebase rebuild)

- Diff `reference/topobench_explain_original/` (once ingested) against
  `src/toposhap/{shapley,cells}` — same picks on a sample of frozen games.
- Reconcile `patches/topobench_patches.py` bugs 2–3 with the original fixes
  from #1's worktree (currently runtime guards only; bug 1 is verified and
  test-reproduced here).
- Reconcile leaf-level schemas in `scripts/make_tables.py::_per_seed_leaf`
  and `toposhap/io/results.py` with the real files (loaders fail loudly by
  design).

## 6. Gap-fill campaign (2026-08-28/29, this cluster)

Everything below ran locally after the 08-18 freeze and EXTENDS the frozen
frames in place; the manifest re-cut (freeze v2) is pending the last arms.

| arm | runner | assembly | lands in |
| --- | --- | --- | --- |
| B=1 selection on MUTAG/NCI109/Alkane/MANTRA | `experiments/run_autok.py --direction backward` | `scripts/backfill_b1_marker.py` (B=2-profile pricing) | `autok_marker.parquet` |
| MANTRA B=1 balanced metrics | — | `scripts/patch_b1_mantra_balanced.py` (checkpoint reload) | raw autok JSONLs |
| post-hoc FLOPs for campaign B=2 runs | `experiments/profile_b2_flops.py` | `scripts/patch_b2_marker_flops.py` | `ladder_b2_marker.parquet` |
| HOPSE selection on Cora/Citeseer | `experiments/run_ladder_b2_hopse.py` (whitelist extended; `--prepare-only` CPU pre-pass first) | run, then EXCLUDED from frames by scope decision 2026-08-29 (HOPSE evaluated on complex-level tasks only; raw JSONLs kept under `results/ladder_b2_hopse/`). Pubmed preprocessing is additionally infeasible (dense-eigh segfault at scale). | — |
| MANTRA GCCN sweep (campaign 23-config grid) | `experiments/run_gccn_sweep.py` (grid read from `cost13.parquet`; balanced metrics in-run) | `experiments/assemble_gccn_sweep.py` | `baseline_markers.parquet` |
| PROTEINS no-ground-truth faithfulness | nogt runners (T3) | `scripts/make_paper_tables.py` reads `results/nogt/` | Table 3 |

Conventions carried over: one selection pipeline, seeds 42–44, measured
FLOPs (post-hoc profiling documented in the paper's FLOPs appendix),
balanced accuracy on MANTRA, selection always on plain validation accuracy.
