#!/bin/bash
# HOPSE ladder B=2 calibration gates, one GPU job. Submit:
#   sbatch experiments/slurm/ladder_b2_hopse_gates.sbatch
# Gate A: full 8-block HOPSE trainings at seeds 42-44 per molecule dataset
#         vs #13's full_menu reference + zeroing mechanism checks.
# Gate B: one full ladder B=2 HOPSE cell (mutagenicity_gxai, seed 42).
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-hopse-gates
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

echo "=== gate A: full-model reference + zeroing mechanism ==="
python "$REPO/experiments/hopse_gates.py"

echo "=== gate B: mutagenicity_gxai seed 42, full ladder B=2 HOPSE cell ==="
python "$REPO/experiments/run_ladder_b2_hopse.py" \
    --dataset mutagenicity_gxai --seed 42
