#!/bin/bash
# MANTRA gates, one GPU job: one end-to-end cell per family + HOPSE
# full-mask no-op. (Param sanity runs on the login node before submission.)
#   sbatch experiments/slurm/mantra_gates.sbatch
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-mantra-gates
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

echo "=== gate: HOPSE full-mask no-op, MANTRA ==="
python "$REPO/experiments/hopse_noop_gate.py" --dataset mantra_orientation

echo "=== gate: mantra_orientation seed 42, GCCN ladder B=2 ==="
python "$REPO/experiments/run_ladder_b2.py" --dataset mantra_orientation --seed 42

echo "=== gate: mantra_orientation seed 42, HOPSE ladder B=2 ==="
python "$REPO/experiments/run_ladder_b2_hopse.py" --dataset mantra_orientation --seed 42
