# Method improvements: stability and efficiency

Analysis date 2026-08-16, grounded in frozen #12 data (`analysis12`,
`cheap_sim12`, `cost_wallclock12`) and the recovered original implementation
(`reference/exp1_worktree/.../topobench/explain/`). Everything here is a
proposal or a verification target — nothing below changes a frozen verdict.
Labels: [now] = analysis-only on existing/#14 data; [cheap] = small
pre-registered run; [post] = post-deadline / camera-ready.

## What the data says the instability IS

1. **Pick-identity instability is universal, value instability is not.**
   Modal-pick agreement at the production budget (128 passes) is 0.28–0.55 on
   every dataset, and only 0.61–0.77 at 512. But the mean |cheap-regret| of
   resampled picks is small and budget-responsive everywhere except the
   committed seeds (benzene 0.074→0.023 from 32→512; fluoride ≤0.0004 from 64
   up). Most disagreeing picks are near-ties in value — the enormous bars come
   only where near-tied picks differ hugely after retraining.
2. **Committed stems are structure, not noise.** Anchored-containing fraction
   of 50 resampled picks at 512 passes: benzene s43 = 0.00 (both k), NCI1
   s43 = 0.00, fluoride s42/s43 = 0.00. No budget fixes these — confirms the
   substitute-commitment diagnosis with the strongest possible signature.
3. **Catastrophes are rare and validation-visible.** Across all 63 #12 cells
   exactly 2 are catastrophic (test >0.04 below their cell median): benzene
   k=2/k=3 seed 43. Both are equally visible in *validation*
   (0.759/0.760 vs ~0.92 siblings). Zero false alarms at the same threshold.
4. **The selectors that “disagree” with guided are often right.** greedy picks
   an anchor-containing mask in 45/63 cells, argmax in 37/63, vs plain top-k’s
   sampled picks at 0–100% depending on seed. On benzene s43 both greedy and
   argmax choose mask 33 (up_adjacency-0 + 2-up_adjacency-0) — the family that
   retrains to ~0.92.

## Stability proposals (ordered by evidence strength)

- **S1 [in flight, #14] Anchored selection.** Already pre-registered; the
  committed-stem data above says it is the right fix for the molecule
  datasets. Caveat the data also raises: fluoride picks anchor-free masks in
  100% of resamples on two seeds *without* any catastrophe — the anchor prior
  is not universally correct, so anchoring must be reported per-dataset, not
  sold as a global rule.
- **S2 [cheap, ~1 line of recipe code] Certify-or-repair wrapper.** After the
  pruned continuation, compare its validation accuracy against a free in-run
  reference (the stem’s full-coalition masked-game value v(full), and/or the
  anchored candidate). If val < reference − δ (δ ≈ 3× training-noise sd,
  0.014), run one fallback continuation (anchored or greedy pick) and keep the
  val-better model. Evidence: catches 2/2 catastrophes with 0/61 false alarms
  in #12 → expected extra cost ≈ (2/63)×continuation ≈ +0.02 runs average,
  bounded by +0.8 runs worst case. Dataset-agnostic (no anchor prior), honest
  (val-only decisions), and it also covers the case where an anchored cell
  lands *below* plain in #14. Implementation: `run_recipe()` in
  `src/toposhap/neighborhoods/stem.py`.
- **S3 [cheap] Multi-stem attribution averaging.** The root cause is the stem
  lottery; average phi over 2–3 independent 5-epoch stems (or intersect their
  top-k). Cost on benzene: +2 stems (68 s each) + 2 sampled games ≈ +0.5 run
  equivalents; kills exactly the variance source the anchoring works around.
  Pre-register as “stability arm”: predicted effect = anchored-fraction of
  benzene-s43-style stems moves off 0 because the average includes
  uncommitted stems.
- **S4 [in flight, #14] Greedy on the game** with the submodularity census
  gating the (1−1/e) claim. The 45/63 anchor rate shows it resolves
  redundancy without being told about chemistry.
- **S5 [now] Tie-aware reporting.** Report value-regret alongside pick
  agreement in the stability section; agreement alone (0.3–0.5) overstates
  instability because most flips are benign near-ties (mean |regret| ≈0.001–0.02).

## Efficiency: the game is the cost center, and it is mis-allocated

Measured components (seconds, #12; game at exact pricing):

| dataset | sweep-run | stem5 | game | continuation | game/run |
|---|---|---|---|---|---|
| benzene | 641 | 68 | 1564 | 526 | 2.4× |
| mutagenicity | 99 | 18 | 279 | 84 | 2.8× |
| NCI1 | 232 | 25 | 889 | 219 | 3.8× |
| fluoride | 529 | 50 | 1099 | 405 | 2.1× |
| cora | 23 | 7 | 304 | 21 | 13.5× |
| citeseer | 38 | 9 | 946 | 32 | 25.1× |
| pubmed | 54 | 9 | 1053 | 38 | 19.4× |

- **E1 [now — verify against #13 Stage D] Pricing arithmetic.** At n=11
  players, 128 complete permutations cost ≈1,400 *unique* coalition
  evaluations (with caching; `cheap_sim12.meta` confirms unique-eval
  accounting) vs 2,047 for the exact game — i.e. ≈0.68× exact, not ~0.1×.
  Either the production sampler cheapens per-eval cost somewhere the #12
  clock didn’t isolate, or the 0.1× star pricing under-charges the game.
  Stage D’s FLOPs-per-run rows settle it; check before the freeze because
  star x-positions use the 0.1 factor.
- **E2 [cheap] Cheaper evaluations, not fewer: subsample the validation set
  with common random numbers.** Marginals are paired differences
  v(S∪{i})−v(S) on the *same* val subset, so per-sample noise largely
  cancels; a fixed 25% subsample cuts per-eval cost ~4× with second-order
  effect on pick quality. Certify step (S2) still runs on full val. Biggest
  effect exactly where the game is most mispriced (citation datasets:
  sampled game alone ≈2.5 full runs on citeseer today).
- **E3 [cheap] Adaptive pass budget.** Stop when the CI on the k-th/(k+1)-th
  Shapley gap resolves (our estimator already returns per-player stderr;
  `SampledAttribution.stderr`). Fluoride is converged by 64 passes while
  benzene still improves at 512 — a uniform 128 wastes compute on one and
  starves the other. Same total compute, reallocated, strictly better.
- **E4 [post] Batched coalition forwards.** The original cell game batches
  block-diagonal replicas (`masking.py:collate_replicated`) but the
  neighborhood game evaluates coalitions serially (`nbhd_game.py:__call__`
  loops). Mask-per-replica route masking evaluates B coalitions per forward.
- **E5 [now, rule] Exact-when-cheap.** If the adaptive sampler’s unique-eval
  count approaches 2,047×(subsample fraction), switch to exact enumeration on
  the subsampled val — same cost, zero estimator variance, and the “exact”
  headline for free at n=11.
- **E6 Non-lever: stems.** Stems are 2–10% of recipe cost; do not shrink
  them. Optional [cheap] probe: stem epochs 5 vs 15 on benzene ×3 seeds,
  outcome = anchored-fraction of resamples (does a longer stem soften or
  entrench commitment?).

## Game-vs-deployment alignment (parity findings, feed into the paper’s honesty)

- **A1 Zero-fill vs removal.** Original masking zero-fills dropped route
  outputs (`nbhd_game.py:_routes_masked`); pruning *removes* routes, and the
  backbone then passes input features through for rank with no live routes
  (`gccn.py` fallback). The two differ exactly on rank-starving coalitions —
  small coalitions, which dominate low-k Shapley weights. Our
  `CoalitionMaskedBackbone` implements removal semantics (matches
  deployment). Quantify on #14 data [now]: recompute picks under both
  semantics from the same stems; if picks differ, removal semantics should
  correlate better with retrained truth (ρ vs the 0.8322 baseline).
- **A2 Normalization drift hypothesis.** Masked-game evaluations feed
  BatchNorm statistics computed on full-ensemble activations; small
  coalitions are off-distribution, biasing v(S) low. Test [now]: regress
  cheap-vs-true residuals on |S| in the #9/#14 landscapes; if confirmed,
  per-coalition BN recalibration (one val pass) or norm-free stems are the
  fix [post].

## Automatic k, scored against exact truth (added after #14 pod ingest)

All four pre-registered rules scored on the #9 exact 6-space landscapes
(12 datasets, rules consuming exact Shapley/exact game — a rule-quality
check, independent of stem noise):

| rule | mean regret | max regret | mean k |
|---|---|---|---|
| greedy-with-noise-stop (0.014) | **0.0119** | 0.0326 | **2.1** |
| threshold-at-zero | 0.0132 | 0.0475 | 5.1 |
| smallest-sufficient (0.014) | 0.0149 | 0.0351 | 2.2 |
| cost-regularized (0.005) | 0.0167 | 0.1072 | 3.8 |
| fixed k=3 (top-k by phi) | 0.0313 | **0.2057** | 3.0 |

Greedy-with-noise-stop wins on every aggregate: lowest regret with the
smallest models, and it self-sizes (k=5 on MUTAG/ZINC where the optimum is
large, k=1 on citations where one route suffices, hitting the exact optimum
on MUTAG/ZINC/ZINC_gin). Fixed k=3 — which looks good on the 7-dataset
marker suite — has a 0.21 catastrophe on MUTAG (redundancy trap in top-k by
phi). The stopping threshold is the *measured* retraining noise, so k stops
being a hyperparameter: it emerges where marginal value crosses the noise
floor. Production-side confirmation (rules consuming the cheap stem game
instead of truth) is #14's battery, pre-registered, pending its finish.

### The stop threshold is practitioner-computable (no oracle noise needed)

The greedy stop needs a noise floor, and true retraining noise requires
repeat runs — but the binomial floor of the practitioner's own validation
set, sqrt(p(1-p)/m), is known from the val size before any training and
matches measured retraining noise in order of magnitude on all 7 marker
datasets (slightly larger on 6/7 — conservative direction). Scored on exact
truth, greedy-stop with the 1-se binomial threshold does as well as the
oracle threshold (mean regret 0.0090 vs 0.0121; it adapts per dataset).
Free refinements if wanted: paired McNemar se from the game's own
evaluations, or val-jitter over the stem's final epochs. Paper subsection:
we can *validate* the proxy because we measured true retraining noise.

### Run 1 of the local auto-k arm (fresh stems, 35/35 cells, 2026-08-16)

Forward greedy with the binomial stop, executed on freshly trained stems
(protocol-matched: parameter counts reproduce #14's to the digit — NOTE the
record's model is `topotune_nonlinear_exact` with NoReadOut linear sum-pool,
not plain `cell/topotune`; game replay corr 0.99998 vs #14's cached games at
matched seed). Result: k=1-2 nearly everywhere, and WORSE than anchored k=3
on most datasets (fluoride 0.848 vs 0.936; cora 0.769±0.081). Diagnosis
(evidenced in results/autok/GATES.md): a 5-epoch stem masked to 1-2 routes is
effectively untrained — small-coalition values collapse to one-class
accuracy, so forward greedy walks the exact region where the cheap game is
uninformative and stops on arbitrary singletons. The rule is sound on the
TRUE game (see exact-truth scoring above); the *direction* of the walk must
match where the cheap game is faithful — near the full ensemble. Fix in
flight: BACKWARD ELIMINATION from the full coalition with the same binomial
stop (remove the cheapest route while removal costs < se). Substitutes are
handled classically (removing a backup is free; the last copy hurts and
stays), and every evaluation happens in the stem's calibrated regime. This
run-1 null is paper material: it demonstrates *why* naive wrapper selection
on attribution games needs the faithful-region argument.

### Run 2: backward elimination (fresh stems, 35/35 cells, 2026-08-16)

Backward elimination with the same binomial stop beats forward on 6/7
datasets, collapses forward's seed variance (cora sd 0.081 -> 0.008), and
matches anchored k=3 within +/-0.007 on 5/7 (beats it on benzene and
mutagenicity). Remaining gaps, honestly: fluoride -0.035 (bimodal — two
seeds tie-cascade to k=1 through zero-cost removals on a flat stem game and
land at 0.83-0.85 while the k=8/9 seeds reach 0.94) and cora -0.020.
Backward keeps larger coalitions (k mostly 7-11): conservative pruning, so
its compute win comes from skipping the sweep, not from a slimmer model.
Same-stem cross-check vs #14 cached games: exact pick reproduction where the
cache covers the path (NCI1 s45, mutagenicity s46; cora s45 exactly on a CPU
stem replay); divergences trace to uncached candidates or GPU stem
nondeterminism. STATUS: exploratory (these local arms are not
pre-registered); the pre-registered #14 battery remains the confirmatory
test. Possible tie-cascade fix if pursued: break zero-cost removal ties by
lowest Shapley value, and/or stop on cumulative-drop > se.

### The reliability tier: the two-candidate race (scored 2026-08-16, no new compute)

Continue BOTH the auto-k backward pick and the anchored k=3 pick from the
same stem; keep the val-better one. Scored post-hoc on the 35 existing cells
(rule consumes val only; test judges it): mean gap to the per-dataset
best-star ORACLE = 0.0000, worst = -0.0036; repairs fluoride (0.901->0.937)
and cora; beats the oracle on NCI1 (+0.005) and mutagenicity (+0.006); sd at
training-noise width on all 7 datasets. Cost ~2.1 training-run equivalents.
This is the practitioner dial's reliable setting: 1.3 runs single-shot
(misses rare and val-visible) / 2.1 runs race (oracle-matching here) /
6 runs menu / sweep for certainty. Exploratory label; #14 battery is the
confirmatory backstop.

### One-run decision rules, scored (2026-08-16)

Guarded one-run auto-k (backward elimination; fall back to anchored k=3 on
the cascade signature k<=2, i.e. the stop never fired): mean -0.0048 / worst
-0.0204 vs the per-dataset oracle, and it fixes fluoride outright (0.9389 —
best of any arm there). Plain anchored-3 scores -0.0018 / -0.0040 on THIS
suite — but that is suite luck: the exact 12-dataset landscapes show fixed
k=3 loses 0.21 on MUTAG (true k*=5, which greedy-stop finds exactly), so a
hardcoded k cannot be the universal one-run answer. The remaining guarded
gap (cora -0.020) is one-stem-irreducible: the stem's own game PREFERS the
large coalition (removals genuinely hurt the young stem); only a
continuation reveals the slim retrained model matches. That missing bit is
exactly what the race's second continuation buys. Candidate one-run signal
to close it (untested hypothesis, post-deadline): pairwise
interaction/substitute mass among kept routes as a consolidation predictor.
NOTE-TO-SELF: executed elimination steps cost < se by construction — trace
flatness audits must use the cascade signature, not step magnitudes (a
first scoring attempt silently degenerated to anchored-3-always).

### THE FROZEN RULE (2026-08-17): the attribution ladder

All prior patches (anchor prior, cascade guard, bespoke race) are RETIRED to
development history — they were iterated against the 7-dataset suite. The
method, frozen before any further scoring:

  Backward elimination on the stem game yields a NESTED LADDER of
  coalitions, one per size. Train B rungs and select by validation. Rung
  choice (fixed 2026-08-17 BEFORE any B>=2 arm ran): rung 1 = the stop rung
  (smallest rung whose next removal costs > binomial se); rungs 2..B = the
  highest-cheap-value rungs at sizes not yet taken. B=1 is the single-shot
  recipe; the menu is B=6 with literature rungs; the sweep is B=all.

Out-of-sample test on the 5 exact-landscape datasets never touched by
today's iterations (MUTAG, NCI109, ZINC, ZINC_gin, alkane_carbonyl): the
ladder CONTAINS the exact global optimum on 4/5 (worst regret +0.0068 on
ZINC_gin); across all 12 landscapes, 10/12 exact-zero, worst +0.0068. The
naive B=1 stop pays up to +0.022 (ZINC) — the measured price of budget 1.
Caveat, disclosed: this validates the rule SHAPE on true games; stem-game
quality is measured by the 7-dataset dev arms and confirmed by #14's
pre-registered battery. Evidence layers kept separate in the paper;
development trajectory disclosed in an appendix.

## Suggested sequencing

1. [now] E1 pricing check + S5 + A1/A2 analyses land with the flat tarball.
2. [cheap, before freeze if #14 leaves room] S2 certify-or-repair — one
   guarded fallback, uses #14’s own anchored-arm data to score the trigger
   post-hoc before any new GPU time.
3. [cheap] E2+E3 in the fresh sampler (both already have hooks:
   `CachedGame.budget`, `SampledAttribution.stderr`).
4. [post] S3 multi-stem, E4 batching, A2 fix — camera-ready ablations.
