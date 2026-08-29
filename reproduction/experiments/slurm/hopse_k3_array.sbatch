#!/bin/bash
# HOPSE fixed-size k=3 (ladder rung) backfill: every HOPSE-family cell with
# a recorded ladder (7 datasets x seeds 42-44 = 21 cells). Submit AFTER the
# MANTRA HOPSE ladder cells exist:
#   sbatch experiments/slurm/hopse_k3_array.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. Throttle %16 (researcher directive); 90-min per-cell cap.
#SBATCH --job-name=toposhap-hopse-k3
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-20%16
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai NCI1 MUTAG NCI109 \
          mantra_orientation)
SEEDS=(42 43 44)
IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

echo "cell: hopse k3 rung dataset=$DATASET seed=$SEED (index $IDX)"
python "$REPO/experiments/run_hopse_k3_rung.py" \
    --dataset "$DATASET" --seed "$SEED"
