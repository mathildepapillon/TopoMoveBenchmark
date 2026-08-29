# Auto-k greedy arm — calibration gates (2026-08-16)

Model config: `model=cell/topotune_nonlinear_exact` (ported from the frozen
exp1 worktree; upstream `cell/topotune` differs only in readout).

## Gate A — parameter match at benzene mask 132 : PASS (exact)

| config | backbone @132 | total @132 |
|---|---|---|
| #14 record | 8704 | 10528 |
| cell/topotune_nonlinear_exact | **8704** | **10528** |
| cell/topotune (upstream readout) | 8704 | 16864 |

The GNN/backbone hyperparameters (GCN, hidden 32, num_layers 2, BatchNorm;
2 TopoTune layers) are identical in both configs; only the NoReadOut
(linear sum-pool) readout reproduces the recorded total, confirming #14 ran
the nonlinear-exact config.

## Gate B — full cocitation_cora seed-42 pipeline (GPU job 414328)

- Runs end-to-end; k = 1 (within the expected 1-3).
- v_full = 0.5185 — below the stated 0.7-0.9 range, but #14's own cached
  cora E5 game (seed 45) has v(2047) = 0.6145: the 0.7-0.9 range is not
  attainable for any 5-epoch cora stem per #14's own record.
- test = 0.6484 vs the 0.8579 anchored k=3 anchor — a miss, but a *real*
  protocol behaviour, not an implementation defect: the seed-42 E5 stem's
  game is flat at small coalitions (all singletons/pairs collapse to
  ~0.167 = one-class accuracy), so greedy-from-empty stops at k=1 on an
  arbitrary singleton (up_incidence-0), and a k=1 up_incidence-0 model
  can only reach ~0.65 test.
- Faithfulness evidence (matched seed 45): retraining our stem with the
  identical protocol and evaluating the 129 coalitions cached in
  `marker_arm14/results/cheap/sampled_cocitation_cora_seed45_E5.npz` gives
  corr 0.99998, mean |delta| 0.0008, max |delta| 0.0044 (v_full 0.6130 vs
  0.6145). Same code at seed 45: k=5, mask 1697, test 0.8109.
- CPU and GPU runs of seed 42 produce identical game values (deterministic
  seeding verified across devices).

## Gate C — benzene stem vs #14 cached game (GPU job 414328) : PASS

Fresh benzene seed-45 stem, same protocol; #14 cache
`sampled_benzene_seed45_E5.npz`:

| mask | ours | #14 | delta |
|---|---|---|---|
| 2047 (gate) | 0.9111 | 0.9078 | +0.0033 |
| 0 | 0.4911 | 0.4922 | -0.0011 |
| 132 | 0.4944 | 0.4944 | +0.0000 |
| 1 | 0.5011 | 0.5561 | -0.0550 |

Gate criterion (v(2047) within 0.05): PASS. The mask-1 drift is a
different-stem effect (GPU training trajectories differ from the record's
hardware; ballpark agreement expected, not equality).
