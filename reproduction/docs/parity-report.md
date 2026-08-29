# Parity report: original leg-1 explainer vs this repo's reimplementation

Date 2026-08-17, before the results freeze. Sources:

- **ORIGINAL** (read-only reference, produced the frozen leg-1 numbers):
  `data/frozen/reference/exp1_worktree/topobench_repo/topobench/explain/`
  — 12 modules; the game is `masking.py:MaskedGCCNGame`, the exact engines
  are `toposhap.py` (receptive-field-decomposed) and `brute_force.py`
  (oracle), the algebra is `interactions.py`, the neighborhood game is
  `nbhd_game.py`.
- **OURS**: `src/toposhap/` (`cells/`, `shapley/`, `metrics/`,
  `neighborhoods/`) and its upstream port on `external/TopoBench` branch
  `toposhap-upstream` (`topobench/explain/`).

Verification: `tests/test_parity_original.py` imports the original modules
directly from the frozen path (sys.path/`sys.modules` injection; the
original's `collate_fn` import resolves against the installed pinned
TopoBench, its `topobench.explain.interactions` import resolves against the
reference copy registered under that dotted name) and checks numerical parity
on the two-triangle fixture. Results at the bottom.

## Semantic diff

"Frozen leg-1 semantics" = what `run_explain.py` ran with its defaults:
`baseline="train_mean"` (per-rank mean **encoded** row over the train split,
`train_split_baseline_rows`), `target="label"` (target-class **logit**),
float64 forwards, masking **encoded** features, exact modes (`local` /
`subcube`) cross-checked against the `brute_force.py` oracle and shapiq.

| # | Axis | ORIGINAL (produced frozen leg-1 numbers) | OURS before this audit | Status after this audit |
|---|------|------------------------------------------|------------------------|-------------------------|
| 1 | Exact engine math | Two independent paths: Moebius/Harsanyi fast subset transform + equal sharing (`interactions.py`), and the marginal-contribution oracle (`brute_force.py`); agreement 2.254e-14 | One path: marginal-contribution definition (`shapley/exact.py:shapley_values`), independent brute force in tests | **Equivalent math.** Verified on the same game values: max dev 3.5e-18 vs the oracle, 5.6e-17 oracle-vs-Moebius (test `test_exact_engines_agree_on_the_same_game`) |
| 2 | Cell-game masking point | **Encoded** features: encoder applied once, coalition rewrites rows fed to the message-passing backbone (`masking.py` lines 133–139, 324–351) | **Raw** features: rows of `batch.x_{rank}` before the encoder | **Aligned.** `CellMaskingGame(encoder=...)` encodes once at construction and masks the encoded rows; raw masking stays the `encoder=None` legacy option. Verified: encoded rows bit-identical, coalition values equal to 8.0e-17 |
| 3 | Cell-game baseline | Rank-dependent rows: default `train_mean` (per-rank mean encoded row over the train split), options `complex_mean` (per-rank mean over this complex) and `zeros` | Zeros, or a per-cell replacement matrix per rank | **Aligned.** `baseline` now accepts `"complex_mean"` (default when an encoder is given), `"zeros"`, per-rank **row** dicts (the `train_mean`/`baseline_rows` convention — feed train-split means for exact frozen semantics), and the legacy per-cell matrices. Verified for all three against the original: max dev 2.2e-16 |
| 4 | Cell-game value scalar | Target-class **logit**, `target="label"` by default (`"predicted"`/index available); v(full) equals the model's own logit (boundary check) | Whatever `model_fn` returns (repo default leans accuracy scoring, exp #8 — a *neighborhood*-game finding) | **Documented, convention preserved.** `model_fn` still defines the scalar; the docstring now states the original convention (target-class logit) and the parity test pins v(full) = model target logit (`test_full_value_is_target_class_logit`). Accuracy scoring remains the deliberate default for the neighborhood games only |
| 5 | Batching | Block-diagonal replicated batches: complex collated once per batch size, only feature rows rewritten, up to 256 coalitions per forward (`collate_replicated`, `_template`) | Serial: one forward per coalition, snapshot/restore around each (TopoTune forward mutates its batch) | **Left as-is (efficiency, not semantics).** Parity at 1e-16 across the full 2048-coalition lattice shows block-diagonal replication is value-neutral. Batched forwards remain the E4 [post] item in [method-improvements](method-improvements.md) |
| 6 | Receptive-field exactness strategies | `local`: linear sum-pool readout decomposes the game into per-readout-cell local games on their receptive fields, cost ≤ Σᵢ2^{\|Rᵢ\|} (the complexity claim); `subcube`: enumerate 2^{\|I\|} over the influential set to *falsify* the trivial-interaction theorem; players outside the influential set provably zero and never enumerated | Full 2^n enumeration when n ≤ `EXACT_PLAYER_LIMIT`=20, permutation sampling above | **Not ported, deliberately.** These are cost decompositions, not different games — the original's own falsification mode (`subcube`) and its brute-force oracle certify they return the same values the plain lattice does. Every exact leg-1 replication in this repo is at or below 20 players, where full enumeration is exact by construction. If leg-1 scaling runs are ever redone here, port `toposhap.py`+`receptive_field.py` rather than re-deriving |
| 7 | Sampled/approximate regime | Lambda-capped KernelSHAP-IQ (shapiq) under a coalition budget — fallback for complexes too large for either exact strategy; frozen *exactness* numbers came from the exact modes | Permutation sampling with antithetic pairing, per-player stderr, unique-eval accounting (`shapley/sampled.py`) | **Different estimators, both approximate; no frozen number depends on either.** Kept ours (it is the production neighborhood estimator with the stability machinery). Not a parity item |
| 8 | Neighborhood game masking | `nbhd_game.py:_routes_masked`: **zero-fill** dropped routes' outputs inside `aggregate_inter_nbhd`; value = target-class logit; serial forwards | `neighborhoods/masking.py:CoalitionMaskedBackbone`: **route-removal** semantics (dropped routes leave the sum; rank with no live route falls back to input features, matching deployment/pruning); accuracy-scored stem game | **Documented divergence, kept ours** — this is the A1 finding in [method-improvements](method-improvements.md) ("Game-vs-deployment alignment"): the two differ exactly on rank-starving coalitions, and removal matches what pruning deploys. Leg-2+ semantics; the frozen leg-1 numbers do not depend on it |
| 9 | Player indexing | Rank-major, cell order within rank; bitmask bit b = player b (`enumerate_coalitions`) | Same: `CellPlayer` list order = bit order; `NEIGHBORHOODS` bit order frozen | **Identical.** Parity test builds players rank-major and compares bit-for-bit |
| 10 | Precision | float64 model + batch by default, "so the alternating Moebius sums are not limited by the model's float32 noise" | dtype of whatever batch/model the caller passes | **Convention documented.** Original mode in the parity test casts both to float64; recommended for exact runs |
| 11 | Mutation guard | `_forward_batch` overwrites `x_{rank}` on the cached template each call | Snapshot **every** rank's features at init, restore in `finally` (CLAUDE.md TopoTune invariant) | **Equivalent effect**, mechanism differs; ours also guards non-player ranks |
| 12 | Metrics | GEA = Jaccard of top-k cells vs ground truth (k = gt count, max over admissible gts); GEF = 1−exp(−KL(full ‖ masked-to-explanation)) (`metrics.py`) | Same formulas (`metrics/faithfulness.py:gea_jaccard`, `gef_unfaithfulness`) plus a random-control floor | **Identical conventions** (GraphXAI-comparable) |
| 13 | Diagnostics | Boundary check, contribution consistency, locality probes, autograd `empirical_dependency`, Moebius-outside-RF theorem check | Executable axiom checks (`metrics/axioms.py`): efficiency, null players, symmetry; batch-invariance guard | **Complementary, overlapping on efficiency/null-players.** The RF-specific diagnostics travel with item 6 if ever ported |
| 14 | Backbone orientation | The frozen worktree ships the **upstream** (dead-route) `interrank_boundary_index` — exp #1 is what *discovered* the dead routes (null players that configs said should matter) | Runtime patch, `TOPOSHAP_INTERRANK_ORIENTATION=fixed` for all attribution runs, `upstream` only for trustworthiness ablations | **Orthogonal to game parity**: both games wrap whatever model they are given. The parity test applies the fixed orientation to the shared model, so parity is checked on the semantics that current runs use |

## What was aligned (src/toposhap/cells/explain.py, mirrored on `toposhap-upstream`)

`CellMaskingGame` (and `explain_cells`) gained:

- `encoder=` — applied exactly once at construction under `no_grad`; masking
  then acts on the **encoded** rows and `model_fn` must run only
  backbone+readout (mirror of `masking.py`'s encode-once).
- `baseline=` extended: `"complex_mean"` (per-rank mean of the post-encoder
  rows — the original's self-contained rank-dependent baseline),
  `"zeros"`, per-rank **row** dict (the original `baseline_rows` /
  `train_mean` convention, broadcast to all masked cells of the rank), or the
  legacy per-cell matrix dict.
- Defaults: **with an encoder, defaults reproduce the original semantics**
  (encoded masking + rank-dependent `complex_mean` baseline; pass train-split
  mean rows for exact `train_mean`); **without an encoder nothing changes**
  (raw masking, zeros) — the reimplementation's behavior stays as the
  explicit legacy option.
- Value convention documented: original = target-class logit
  (`target="label"`); accuracy scoring remains the neighborhood-game default
  (exp #8).

## What was left as an option / deliberately not changed

- Raw-feature masking with zeros: `encoder=None` (unchanged default without
  an encoder).
- Serial evaluation (no block-diagonal batching): value-neutral, tracked as
  E4 in [method-improvements](method-improvements.md).
- `local`/`subcube` receptive-field strategies and the lambda-cap estimator:
  not ported (see rows 6–7).
- Neighborhood-game route-removal vs zero-fill: kept removal (deployment
  semantics), documented as A1 in
  [method-improvements](method-improvements.md).

## Numerical parity (tests/test_parity_original.py, two-triangle fixture, 11 players, float64)

Direct import of the original worked; no oracle literals needed. Both games
share one model (full 11-neighborhood TopoTune backbone, linear per-rank
encoder, sum-pool linear readout, fixed orientation), target = label logit.

| Check | Max abs deviation | Bar |
|---|---|---|
| Encoded feature rows (ours-with-encoder vs original `h_r`) | 0.0 | — |
| 20 random coalitions, `complex_mean` baseline (our default with encoder) | 8.0e-17 | 1e-6 |
| 20 random coalitions, `zeros` baseline | 5.6e-17 | 1e-6 |
| 20 random coalitions, explicit per-rank rows (`train_mean` convention) | 2.2e-16 | 1e-6 |
| Full 2048-coalition lattice, original vs ours-in-original-mode | 1.7e-16 | 1e-6 |
| Exact Shapley, original oracle vs original Moebius path (same values) | 5.6e-17 | 1e-12 |
| Exact Shapley, original oracle vs our engine (same values) | 3.5e-18 | 1e-12 |
| Exact Shapley, cross-implementation end-to-end (11 players) | 2.8e-17 | 1e-6 |
| Efficiency residual (ours) | 2.8e-17 | 1e-12 |
| v(full) − model target logit (both games) | 0.0 | 1e-9 |

Control: the legacy raw-masking game agrees with the original at the unmasked
point and diverges under masking (`test_legacy_raw_masking_is_a_different_game`)
— the encoded-vs-raw masking point is a real semantic difference, which is why
the option switch exists.

## Unresolved / out of scope

- `checkpoints.py`, `storage.py`, `run_explain.py`, `run_nbhd_explain.py`:
  run infrastructure tied to the old cluster's Hydra runs; nothing to align
  (INGEST.md governs what gets copied vs recomputed).
- Exact `train_mean` rows for real datasets require the train split of the
  original runs; the convention (per-rank mean encoded row) is implemented
  and verified with explicit rows — recomputing the actual rows is a data
  ingest question, not a semantics one.
