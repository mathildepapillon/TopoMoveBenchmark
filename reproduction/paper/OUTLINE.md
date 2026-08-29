# TopoSHAP — ICLR 2027 outline (9pp main text)

Register: TopoTune. Rules: no text walls (2–4 sentence paragraphs), no
inline number chains (numbers live in tables/figures), bold-first-
sentence result paragraphs, one claim per paragraph, one running
example reused in every conceptual figure, TopoExplorer rank colors
everywhere (rank 0 #3890D4, rank 1 #C47F9C, rank 2 #87003B).
Accessibility contract: a reader who knows neither Shapley values nor
TDL can follow every figure caption alone.

TERMINOLOGY (field-standard, per TopoTune + HOPSE — binding):
- The domain is the combinatorial complex (V, C, rk): cells, ranked by
  rk: C -> Z>=0; "k-cell" / "cells of rank k". CCs subsume the
  discrete topological domains of TDL — graphs, simplicial complexes,
  cellular complexes, hypergraphs. Players are CELLS OF A CC; nothing
  in the method presumes rank-2 cells exist (hypergraphs have
  set-type relations, no faces).
- Chemistry appears ONLY as TopoTune-style parenthetical gloss, once:
  "atoms (nodes, i.e., cells of rank zero), bonds (edges, i.e., cells
  of rank one), rings (cells of rank two)". Never as the names.
- A neighborhood is a FUNCTION N: C -> P(C) assigning neighbor cells:
  up/down incidence, up/down adjacency (HOPSE: adjacencies A,
  incidences I). "Neighborhoods", never "relations".
- Models: topological neural networks (TNNs) as umbrella; the
  higher-order message passing (HOMP) mechanism; CCNNs; GCCNs
  (generalize and subsume CCNNs) via strictly augmented Hasse graphs
  G_N with per-neighborhood networks omega_N; HOPSE = the
  message-passing-free family (Hasse decompositions + precomputed
  positional/structural encodings).
- Motivation phrases of record: "multi-way interactions",
  "higher-order interactions/structure" — pairwise edges cannot
  capture them.
- Running example: a molecular CC (used with the gloss above), chosen
  because benchmarks live there; every claim stated in rank terms.

Page budget: 1.25 intro | 1.5 background | 2 method | 3.25 results |
0.5 related | 0.5 conclusion+limitations. Appendix unlimited.

---

## §1 Introduction (1.25 pp, includes Fig 1)

Paragraphs (one idea each):
1. TNNs make predictions from higher-order structure — cells of rank
   two and above, multi-way interactions no edge can carry — yet every
   post-hoc explainer speaks only of nodes and edges. (Hook: the
   unit-of-explanation mismatch; molecule gloss here, once, in the
   TopoTune parenthetical style.)
2. We define one cooperative game on the cells of a combinatorial
   complex and read two answers from it: per-cell attributions, and
   the smallest sufficient sub-complex. Because CCs subsume graphs,
   simplicial and cellular complexes, and hypergraphs, one method
   covers all TDL domains — and reduces to a graph explainer at
   rank 1.
3. Three things this buys (the three claims, one sentence each):
   matches/outperforms GNN explainers on their own benchmarks at a
   fraction of the cost; explains topological models and data where
   those explainers are undefined; and, with neighborhoods as players,
   selects architectures for a tenth of a sweep's cost.
4. A finding, honestly framed: explanation quality is a property of
   the SUBJECT — TopoSHAP's own diagnostics locate the cause (feature
   caches in the standard lift) and the fix (the structural lift)
   raises accuracy and explainability together.
5. Contributions list (game + two outputs; reduction proposition;
   fair benchmark campaign; lift finding + fix; selection recipe;
   open-source pipeline).

**Figure 1 (overview, full width):** left = a combinatorial complex
(a lifted molecular CC, ranks named by color: rank-0/1/2 cells);
center = the participation game v(S) with a detached coalition drawn;
right = the two outputs (φ bars on cells; the sufficient sub-complex
highlighted); bottom strip = the three claim panels, one headline
number each. Every later figure is a zoom into a region of Fig 1.

## §2 Background (1.5 pp) — teach both audiences in pictures

2.1 *Combinatorial complexes without formalism.* One figure-first
pass: the domains strip (graph, simplicial complex, cellular complex,
hypergraph — all subsumed by the CC), then the lift (graph → cellular
CC), ranks named by color. Definition box (V, C, rk) AFTER the
picture, TopoTune's wording. Two one-sentence anchors: graphs = the
rank-<=1 special case (powers §4.1); hypergraphs = set-type relations
with no rank-2 cells (the method never presumes faces exist).

2.2 *Models: TNNs and neighborhoods.* A neighborhood is a function
N: C -> P(C); the four families drawn as labeled arrows on the same
CC (up/down incidence, up/down adjacency). HOMP in one sentence; the
GCCN layer as a picture (per-neighborhood strictly augmented Hasse
graphs G_N, networks omega_N, aggregation), the one layer equation;
HOPSE named as the message-passing-free family — the game treats both
identically because it only needs forward evaluations. The
11-neighborhood vocabulary listed once.

2.3 *Shapley values from one sentence.* "Pay each player its
contribution averaged over every order the team could have assembled
in" — then the formula in a box, then the four axioms as one line
each with plain-language glosses (efficiency = credits sum to the
prediction; null player = ignored cells get zero — the axiom that
later catches models ignoring their topology).

**Figure 2 (two-row conceptual):** row 1 = the domains strip (graph,
simplicial complex, cellular complex, hypergraph, all as CCs; ranks by
color) with the lift as the graph→cellular arrow; row 2 = a 4-cell toy
game: coalitions as filled/hollow cells, v(S) per coalition, one
cell's marginal contributions averaged → its φ. This is the joint
TDL + Shapley tutorial for the naive reader.

## §3 Method (2 pp) — one game, two outputs, two class rules

3.1 *The participation game* (current §4.1 text, tightened): v(S) =
model's score for the explained class on the induced sub-complex;
detachment = features out, every neighborhood matrix rebuilt, pooling
restricted. One paragraph on why detachment rather than zeroing
(forward-reference the leak quantified in §5.4).

3.2 *Two outputs.* Attribution (φ, axioms, P·n cost) and explanation
(greedy top-down prune, (n²−b²)/2 cost, nested budgets). One paragraph
each + the cost line. Necessity vs sufficiency in one sentence
(necessity-dominant vs sufficiency-dominant benchmarks, no numbers).

3.3 *The explained class: two instantiations of one rule.*
v = the model's score for the explained class. Benchmarks with
annotated motifs: the ground-truth class's logit. No labels (deployed
use): the margin toward the model's own prediction. Same game
otherwise; this is the section header the reader remembers.

3.4 *Neighborhoods explain performance.* The same template with
players = the 11 neighborhoods and v = validation accuracy; recipe box
(exists). Two sentences on honest pricing (sampled game ≈ 1/10 sweep;
exact-game pricing refuted, appendix).

3.5 *Relation to graph explainers* (current §4.1 paragraph): the
rank-1 reduction (SubgraphX = coalition search on this game with
zero-fill + MCTS; both substitutions ablated in §5.1); GNNExplainer/
PGExplainer = relaxed masks, no game. Proposition (reduction) stated,
proof in appendix.

**Figure 3 (method):** the detachment pipeline on the running CC —
one rank-1 cell detached → its incidence rows/columns masked, every
neighborhood matrix rebuilt → pooling excludes it; then the two
outputs side by side on the same complex. (Adapted from the
structural-lift artifact drawings.)

## §4 Results (3.25 pp) — bold-first-sentence paragraphs throughout

### 4.1 Graph benchmarks: the rank-1 validation (0.9 pp)
- **TopoSHAP matches or outperforms every GNN explainer on the
  GraphXAI benchmarks.** Table 1: attribution + explanation rows vs
  GNNExplainer/PGExplainer/SubgraphX/random on identical GNNs, splits,
  full populations, 3 seeds; fluoride tie stated in the caption.
- **The explanation output costs ~1/500 of SubgraphX per instance.**
  Cost column in Table 1 (measured seconds + eval counts).
- **Both removal semantics and search direction matter.** One
  paragraph: zero-fill ablation reproduces SubgraphX-on-GCN; bottom-up
  growth fails (OOD trap). Numbers in Table 1's ablation block or
  appendix table.
- **Where a baseline wins, the game says why.** The alignment gap
  Δ = v(S*) − v(S_gt): fluoride's GIN provably classifies from
  non-motif evidence (fig panel).

### 4.2 Topological models and data (1.5 pp) — the contribution
- **The standard lift caches structure into features, and models read
  the cache.** The mechanism paragraph + Fig 4 (the structural-lift
  two-convention figure, already designed): ProjectionSum vs constant
  features; composition evidence (the rank-2 cell is kept while its
  boundary rank-1 cells are dropped).
- **The structural lift makes GCCNs more accurate AND more
  explainable.** Table 2 (lifted benchmarks): structural full-vocab
  GCCN vs cached, both outputs, cell + node GEA, alignment ceiling
  column; accuracy row. The one-sentence practitioner rule.
- **On topological data without ground truth, TopoSHAP is the most
  faithful method and reports how much topology a model uses.**
  NCI1/PROTEINS deletion-AUC table + rank-share diagnostic; insertion
  curves in appendix. (No-GT = the predicted-class instantiation.)
- **On native topology (MANTRA), rank profiles separate mechanisms.**
  The multipair figure: per-rank credit for CWN/ladder/anchored/HOPSE
  with the vertices-only CWN caught by the null-player axiom;
  deletion-AUC vs baselines.

### 4.3 Performance explanations (0.5 pp)
- **One stem training plus attribution replaces a sweep.** Recipe
  results (frozen arm-2 campaign): selection quality vs sweep at
  sampled-game pricing; Pareto figure (exists). k=6/k=3 dataset notes
  per the framing rules.

### 4.4 Trustworthiness (0.35 pp)
- **The axioms are executable tests, and they caught real bugs.**
  Pool-faithful leak (quantified), rebuild guard catches, the
  feature-cache finding as the third instance; exactness vs brute
  force at machine precision. Honest-nulls sentence.

## §5 Related work (0.5 pp)
GNN explainers (perturbation/search family incl. SubgraphX, Zorro;
learned masks; gradient methods) — positioned via the reduction, cited
generously. Shapley in ML (SHAP, sampling estimators, interactions).
TDL (TopoBench/TopoTune/CWN/HOPSE/MANTRA). Explainability-for-
architecture (NAS-adjacent) for arm 2.

## §6 Conclusion + limitations (0.5 pp)
One game, two outputs, three uses. Limitations, honest: n²-scale
search needs k-hop localization on large graphs; GEA on lifted
domains measures model–motif alignment (our ceiling separates it);
value-function rule has two instantiations; complexes evaluated at
benchmark scale (tens to hundreds of cells); hypergraph domains are
covered by the formalism but not yet evaluated empirically.

---

## Figures inventory (7 main-text)
F1 overview (new; from framework artifact) · F2 lift+Shapley tutorial
(new) · F3 method/detachment (adapt structural artifact) · F4
structural lift mechanism (exists as artifact) · F5 alignment-gap
distributions (new, from oracle data) · F6 MANTRA rank profiles
(exists in deck) · F7 arm-2 Pareto/recipe (exists in deck).
Tables: T1 graph benchmarks (+cost), T2 lifted benchmarks (+ceiling),
T3 no-GT faithfulness. All via make_tables.py macros.

## Appendix map
Proofs (T1 axioms, reduction prop, null-player); the k-arm sweep +
route-lottery investigation; lc semantics ablation; zero-fill
ablation details; probe ladder; estimator variance; experimental
details + compute accounting; glossary (exists).
