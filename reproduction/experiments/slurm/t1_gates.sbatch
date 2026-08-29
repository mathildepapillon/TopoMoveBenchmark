#!/bin/bash
# Tier-1 coverage gates, one GPU job. Submit:
#   sbatch experiments/slurm/t1_gates.sbatch
# Gate (a) runs on the login node (CPU param count) before submission.
# Gate (b): full MUTAG seed-42 cells for both GCCN arms.
# Gate (c): HOPSE full-mask no-op exactness on MUTAG and NCI109.
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-t1-gates
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

echo "=== gate (b) part 1: MUTAG seed 42, anchored k=3 ==="
python "$REPO/experiments/run_anchored3.py" --dataset MUTAG --seed 42

echo "=== gate (b) part 2: MUTAG seed 42, GCCN ladder B=2 ==="
python "$REPO/experiments/run_ladder_b2.py" --dataset MUTAG --seed 42

echo "=== gate (c): HOPSE full-mask no-op, MUTAG + NCI109 ==="
python "$REPO/experiments/hopse_noop_gate.py" --dataset MUTAG
python "$REPO/experiments/hopse_noop_gate.py" --dataset NCI109
