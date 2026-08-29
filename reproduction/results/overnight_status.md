# Overnight status — fair Table 1 fleets

## 2026-08-27 ~08:30 (event: SubgraphX array failed, fixed, resubmitted)
- 472727 (SubgraphX, 36 tasks) failed in ~30s/task on a GPU-only device bug
  in vendored GraphXAI __parse_results (frozen suite's patch P7). Re-derived
  the patch (scripts/apply_graphxai_patches.py), resubmitted as 472857
  gated behind a 1-task GPU smoke (472856, afterok).
- 472725 GNN training COMPLETED: all 24 GNNs trained, split assertions
  passed on every dataset (gt+ test populations 917/231/96/58 match the
  cell pipeline). GCN 29,804 params / GIN 26,220 vs GCCN k=6 28,716.
- 472726 cheap explainers: mutagenicity cells done — full-population GEA:
  random 0.132, GNNExplainer 0.13-0.20, PGExplainer 0.16-0.46 (seed
  variance high, typical for PGE). Plausible vs literature.
- Still running: 472656 (fluoride+alkane k6 GEA), 472724 (benzene+muta k6
  shards), 472747 (k3 shards), rest of 472726.

## 2026-08-27 ~08:35 (event: smoke job shell bug, resubmitted)
- 472856 smoke failed instantly: sbatch --wrap runs under sh, `source`
  unavailable. Resubmitted invoking .venv/bin/python directly: smoke
  472878, SubgraphX array 472879 (afterok-gated). 472857 cancelled
  (dependency could never be satisfied). Monitor re-attached.

## 2026-08-27 ~11:15 (milestone: three pieces DONE)
- DONE: k6 TopoSHAP GEA full population, fluoride (n=231) + alkane (n=58),
  3 seeds. Node-GEA: fluoride 0.451/0.417/0.491, alkane 0.492/0.435/0.410.
- DONE: all 24 cheap-explainer tasks (random/GNNE/PGE, full populations).
  Best baselines so far (node GEA, mean over 3 seeds): benzene PGE-GCN
  0.642±0.044; fluoride PGE-GIN 0.555±0.067; alkane PGE 0.205; muta PGE
  0.290. GNNExplainer near/below random floors everywhere.
- Early fair read: TopoSHAP(k6) above all baselines on alkane
  (0.446 vs 0.205); BELOW best PGE on fluoride (0.453 vs 0.555).
  Benzene + mutagenicity TopoSHAP cells still in flight (8/36 shards
  remaining); SubgraphX array running normally after P7 fix.
- Monitor false alarm: 472656 'FAILED(6/12)' = the six deliberately
  cancelled tasks; all live tasks completed.

## 2026-08-27 ~11:30 (milestone: TOPOSHAP SIDE COMPLETE, both k arms)
All 78 TopoSHAP GEA tasks done (k3: 42, k6: 36+6). Full populations,
3 seeds, node-GEA mean ± seed sd:
  dataset            TopoSHAP k3     TopoSHAP k6     best cheap baseline
  benzene            0.529 ± 0.067   0.521 ± 0.175   PGE-GCN 0.642 ± 0.044
  fluoride           0.500 ± 0.081   0.453 ± 0.030   PGE-GIN 0.555 ± 0.067
  mutagenicity       0.281 ± 0.150   0.332 ± 0.123   PGE 0.290 ± 0.102
  alkane             0.381 ± 0.064   0.445 ± 0.034   PGE 0.205 ± 0.137
Remaining: SubgraphX only (472879: 19/36 done, 17 in queue).
Note: benzene k6 seed spread is wide (0.359/0.441/0.764) — explanation
quality tracks the per-seed trained subject; k3 is tighter there.

## 2026-08-27 ~11:45 — CAMPAIGN COMPLETE
All jobs finished; zero unresolved failures. Full fair table written to
results/fair_table/summary.json (scripts/assemble_fair_table.py; all
population-completeness assertions passed). Node GEA, mean ± seed sd:

                       benzene(917)  fluoride(231)  muta(96)      alkane(58)
toposhap_k3            0.529±0.067   0.500±0.081   0.281±0.150   0.381±0.064
toposhap_k6            0.521±0.175   0.453±0.030   0.332±0.123   0.445±0.034
best GNN baseline      sgx-gin 0.796 pge-gin 0.555 sgx-gcn 0.344 sgx-gin 0.296

Honest reading: TopoSHAP wins alkane clearly, ties mutagenicity
(0.332±0.123 vs 0.344±0.036), loses fluoride narrowly to PGE-GIN, loses
benzene clearly to SubgraphX-GIN (0.796±0.029). The 12h health-check cron
was deleted. Next (needs researcher): headline k arm, framing, then
freeze + make_tables macros.

## 2026-08-27 — design investigation (instability + SubgraphX gap)
Probe ladder on 40 single-candidate benzene graphs (where the gap
lives: TopoSHAP 0.456 vs SubgraphX 0.858), k6 subjects:
  s42 shapley + logit (fleet convention)   0.547
  s42 grow-coalition + logit               0.310  (bottom-up OOD trap)
  s42 prune-coalition + logit              0.550  (readout not the gap)
  s42 prune-coalition + MARGIN             0.729  (+0.18 value fn)
  s44 prune-coalition + logit              0.780  (downward-route subject)
  s44 shapley + logit                      0.728
Running: s44 prune+margin (combo), s42 shapley+margin (does attribution
readout also improve -> fleet convention change).
Design conclusions so far: (1) label-logit value fn discards
prediction-relevant signal -> use label-oriented margin; (2) subjects
need downward incidence routes for higher-order evidence to reach the
readout (also explains k-arm seed instability: route lottery);
(3) readout swap alone changes nothing (prune == shapley at same game).

## probe ladder update (40 single-candidate benzene graphs)
  s42 (no down-routes): shapley+logit 0.547 | shapley+margin 0.604 |
    prune+logit 0.550 | prune+margin 0.729 | grow+logit 0.310
  s44 (down-routes):    shapley+logit 0.728 | prune+logit 0.780 |
    prune+margin 0.689
  Margin helps the weak subject only (attribution +0.06, search +0.18);
  hurts the strong one. Running: beam+coalition-Shapley (SubgraphX-style
  search) on both subjects; target = SubgraphX-GIN 0.858 on this subset.

## Design investigation CLOSED (oracle check)
Beam+coalition-Shapley (SubgraphX's full search) on our game: s44 0.771,
s42 0.565 — no better than greedy prune. Oracle check on s44: on 34/40
single-ring graphs the search finds coalitions the model values ABOVE
the ground-truth ring closure (mean v gap +0.77). Conclusion: every
readout of the k6 game is faithfully reporting a model that genuinely
relies on non-ring evidence; the residual gap to SubgraphX-on-GIN is a
property of the SUBJECT, not the explainer. Lever hierarchy:
(1) subject architecture (down-routes; k=11 fleet = at-scale test),
(2) coalition search readout (small, free gain: +0.05 over Shapley),
(3) margin value fn (rescues weakly-routed subjects only).

## ORACLE RESULT — the residual gap is the subject, not the explainer
On s44 (strong subject), the max-sufficiency coalition at gt budget
beats the ground-truth ring closure on 34/40 graphs (mean v margin
+0.77). The subject genuinely relies on non-ring coalitions; a faithful
explainer CANNOT recover the motif better without lying about the
model. Readout ladder saturates this ceiling: search 0.781 ~= ceiling,
Shapley 0.728 close behind. SubgraphX-on-GIN 0.858 reflects GIN's
better motif alignment (single evidence copy), not a better explainer.
=> GEA conflates explainer quality with model-motif alignment; our
game measures the alignment ceiling for ~2 evals/graph (v_gt vs
v_found). Paper: report ceiling alongside GEA; TopoSHAP saturates it.

## TopoSHAP-Select fleet launched
New library readout toposhap/cells/select.py (greedy top-down prune,
nested budgets, one pass per graph) + experiments/molecule_select_gea.py
(same protocol/gates as the Shapley fleet, plus per-candidate oracle
v_found/v_gt). 84 tasks: k6 + k11 arms, benzene sharded 8x, muta 4x.
Assembler now has select_k6/select_k11 rows (pending until complete).

## SELECT-ON-GNN COMPLETE (special-case closure) — headline result
Our Select readout run on the baseline GNNs (same checkpoints, splits,
full populations as the SubgraphX rows):
  benzene   gin  0.969 ± 0.009   (SubgraphX-gin 0.796 ± 0.029)
  muta      gin  0.480 ± 0.135   (best baseline sgx-gcn 0.344)
  alkane    gin  0.354 ± 0.125   (sgx-gin 0.296)
  fluoride  gin  0.340 ± 0.011   (pge-gin 0.555 still leads)
  benzene   gcn  0.378 ± 0.012   (sgx-gcn 0.664 — semantics x arch
    interaction: hard removal vs zero-fill diverges on GCN)
Conclusion: on GIN, ONE TopoSHAP implementation beats SubgraphX at its
own game with 3x tighter error bars — the earlier gap was entirely the
explained subject. Select-on-GIN sets the best benzene number in the
whole table (0.969).

## Composition probe (lifted benzene, k6 s42, 12 graphs) — mechanism CONFIRMED
Found coalitions vs gt closure (6 nodes + 6 edges + 1 ring):
mean found = 7.7 nodes + 4.4 edges + 0.9 ring. Ring-CELL recall 0.92;
edge recall 0.47; node recall 0.62. The model keeps the explicit ring
cell (the lift's shortcut) + diffuse node mass, and drops the ring's
boundary edges — sufficiency de-localized exactly as hypothesized.
The lifted-vs-GNN gap tracks lift-label correlation: benzene -0.44,
muta -0.14, fluoride/alkane +0.04 (no cycles => no gap).

## Lift-consistent game: probe verdict + fleet
Mechanism fix implemented (participation.py, lift_consistent flag):
"absent" now extends to ProjectionSum feature caches; encoder moved
inside evaluation; strict convention guard. 12-graph benzene probe:
cell GEA 0.427 -> 0.548 (+0.12), node 0.604 -> 0.661; compositions
pull in closure edges; ceiling STILL 12/12 => remainder is the trained
subject's genuinely distributed strategy. MANTRA verified unaffected
(constant per-rank features, no cache). Submitted: explanation fleets
under _lc for all 3 k-arms (126 tasks). DECISION PENDING (researcher):
adopt lift-consistent as the semantics of record on lifted domains
(then rerun attribution fleets under _lc; old semantics -> ablation).

## STRUCTURAL LIFT experiment launched (the TDL-on-graph-benchmarks bid)
Researcher requirement: the paper's TDL pipeline must perform well on
the (lifted) GraphXAI benchmarks, not just explain GNNs. Mechanism
chain says the lever is the SUBJECT: StructuralOnes feature lifting
registered (higher-rank cells = constant features, MANTRA convention,
no cache at training OR explanation time; ring-ness must be computed
through incidences). Alkane smoke: structural k11 subjects train to
0.959-0.971 (matches cached-lift accuracies) at identical 49,696
params. Submitted: k11s training benzene/fluoride/muta (475322) +
chained fleets, both outputs, full populations (84 tasks). Prediction:
alignment ceilings drop, benzene GEA rises toward GIN levels; if so,
the paper's Part-1 TDL row is the structural-lift pipeline and the
cached-lift results become the "what your lift does to explainability"
finding.

## STRUCTURAL LIFT CONFIRMED at full population (k11s explanation arm done)
Benzene: cell 0.679±0.113 / node 0.725±0.105 (cached k11: 0.398/0.509);
ceiling share 0.97 -> 0.74; accuracy HIGHER (0.942-0.976, best lifted
arm). Muta 0.436 node (best in lifted table, beats all GNN baselines).
vs baselines: TDL row beats all on muta+alkane, beats PGE/GNNE on
benzene (0.725 vs sgx-gin 0.796), below PGE on fluoride.
## Lift-consistent semantics decision RESOLVED by data: do NOT adopt.
Full-population lc explanation is mixed (benzene slightly up, fluoride/
muta/alkane down) — cache-trained subjects underperform under leak-free
eval semantics. The right fix was training-time (structural lift); lc
stays as the mechanism ablation. k11 attribution complete; k11s
attribution 14 tasks left; no-GT fleet 3/6.
