# MANTRA + HOPSE-k3 campaign status (submitted 2026-08-17 ~14:20 UTC)

**FINAL 2026-08-17 19:51 UTC: `results/CAMPAIGN_DONE` says `status: COMPLETE`.**
All 6 parquets at full row counts, balanced-accuracy fields everywhere,
class-balance record present, no missing cells. Completion path: balanced
eval 417403 timed out at 90 min with one cell left (menu:479); resume
417837 + assembly 417838 finished it. The remaining sweeper passes will
see COMPLETE and no-op.

The whole remaining DAG is on the cluster: it completes even if every
orchestrator dies. Any future session: read this file (job DAG + rescue
commands) and `results/CAMPAIGN_DONE` (assembled/missing inventory, written
by the final assembly job; `status: COMPLETE` vs `PARTIAL`).

## Already done before this submission (do not redo)

- MANTRA gates job 415475 (COMPLETED 06:58): HOPSE full-mask no-op gate
  passed (logit gap 9.5e-07, drop-block bites 8.6e-01); GCCN ladder B=2 and
  HOPSE ladder B=2 seed-42 MANTRA cells on disk.
- Tier-1 + molecule arms: `data/frozen/{anchored3_exact,ladder_b2_marker,
  ladder_b2_hopse,anchored3_exact_t1,ladder_b2_t1,ladder_b2_hopse_t1,
  hopse_sweep_t1}.parquet`; `results/flops_local/`.
- `results/hopse_sweep_mantra/prep_*.json`: 12 preprocessing caches +
  clocked preprocessing_seconds (the sweep RUNS were not done; they are in
  job 417289).

## Job DAG

| job | what | produces |
|---|---|---|
| 417289 `toposhap-mantra` array 0-26%16 | idx 0-2 GCCN ladder seeds 42-44; 3-5 anchored k=3; 6-8 HOPSE ladder; 9-20 HOPSE sweep cfg00-11; 21-26 GCCN menu masks {21,26,72,129,329,479}. Idx 0 and 6 skip instantly (gate cells exist). | `results/{ladder_b2,anchored3_exact,ladder_b2_hopse,hopse_sweep_mantra,gccn_menu_mantra}/*mantra*` |
| 417290 `toposhap-hopse-k3` array 0-20%16, afterany:417289 | fixed-size k=3 (ladder rung) for every HOPSE ladder cell (7 datasets x 3 seeds) | `results/hopse_k3_rung/k3_*.jsonl` |
| 417403 `toposhap-mantra-balanced` (GPU), afterany:417290 | balanced-accuracy backfill for every MANTRA cell (coordinator directive 2026-08-17): reload each saved best checkpoint, eval val+test, patch `val/test_balanced_accuracy` into JSONLs add-only; class-balance record | patched MANTRA JSONLs, `results/mantra_prepare_logs/class_balance.json` |
| 417404 `toposhap-campaign-assemble` (CPU), afterany:417403 (replaces cancelled 417291) | assemble_mantra + assemble_hopse_k3 (`--allow-missing`), writes marker; audits balanced fields | `data/frozen/{anchored3_exact_mantra,ladder_b2_mantra,ladder_b2_hopse_mantra,hopse_sweep_mantra,gccn_menu_mantra,hopse_k3_rung}.parquet`, `results/CAMPAIGN_DONE` |
| 417405 / 417406 `toposhap-mantra` array 3-5 (417406 afterany:417405) | anchored3 MANTRA rescue: the original cells cannot fit the 90-min cap (764 game evals x ~7 s/eval); the game is now journal-resumable (commit 4c799aa), so attempt 417405 journals ~85 min of evals, attempt 417406 replays and completes. 417289_3-5 were cancelled (own jobs, provably doomed). | `results/anchored3_exact/anchored3_mantra_orientation_seed*.jsonl` |
| 417292 / 417293 / 417409 `toposhap-campaign-sweeper` (CPU, begin +2h / +4h / +8h) | self-heal passes: exit if CAMPAIGN_DONE says COMPLETE or campaign jobs still in queue; else resubmit exactly the missing cells (from `experiments/campaign_missing.py`), balanced-eval stage, chained into a fresh assembly job | rescue submissions as needed |

All runners are idempotent-resumable: they skip cells whose final JSONL is
complete, and the multi-seed runners (sweep, menu) journal per-seed rows to
`<out>.jsonl.partial` so a 90-min kill only costs the unfinished seeds.

## Rescue commands (if a stage died and both sweepers are gone)

- Which cells are missing:
  `python experiments/campaign_missing.py --verbose`
- MANTRA arms (any subset; completed cells skip):
  `sbatch --array=IDXLIST%16 experiments/slurm/mantra_arrays.sbatch`
- k3-rung arm: `sbatch --array=IDXLIST%16 experiments/slurm/hopse_k3_array.sbatch`
- Balanced backfill (check: `python experiments/mantra_balanced_eval.py --check`):
  `sbatch experiments/slurm/mantra_balanced_eval.sbatch`
- Assembly + marker: `sbatch experiments/slurm/campaign_assemble.sbatch`
- Everything at once (the sweeper does exactly the above, in order, with
  dependencies): `sbatch scripts/campaign_sweeper.sh`
- Wait on jobs: `monitor_jobs JOB1 [JOB2 ...]` (never sleep/sacct loops).

## After CAMPAIGN_DONE says COMPLETE

Report per handoff: per-dataset mean±sd of every new arm vs baselines
(MANTRA arms vs hopse_sweep_mantra / gccn_menu_mantra val-best; k3-rung vs
ladder B=2 and sweep best), the MANTRA cross-family contrast (GCCN
orientability vs HOPSE — report as measured; expectation on record is GCCN
near the counting ceiling on balanced metrics, HOPSE clearing it),
completed/failed cell counts, and a no-estimated-quantities audit per
parquet. Note: MANTRA monitors val/f1; rows record plain accuracy alongside
(disclosed stand-in — see commit a102448).
