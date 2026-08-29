#!/bin/bash
# Tier-1 HOPSE arms: ladder B=2 (MUTAG/NCI109 x seeds 42-44 = indices 0-5)
# plus the 12-config sweep baseline (indices 6-29: one job per
# (dataset, config), 3 seeds inside each). Submit:
#   sbatch experiments/slurm/t1_hopse_arrays.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. Throttle %16 (researcher directive); 90-min per-cell cap.
#SBATCH --job-name=toposhap-t1-hopse
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-29%16
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

DATASETS=(MUTAG NCI109)
SEEDS=(42 43 44)
IDX=${SLURM_ARRAY_TASK_ID}

if [ "$IDX" -lt 6 ]; then
  DATASET=${DATASETS[$((IDX / 3))]}
  SEED=${SEEDS[$((IDX % 3))]}
  echo "cell: hopse ladder_b2 dataset=$DATASET seed=$SEED (index $IDX)"
  python "$REPO/experiments/run_ladder_b2_hopse.py" \
      --dataset "$DATASET" --seed "$SEED"
else
  J=$((IDX - 6))
  DATASET=${DATASETS[$((J / 12))]}
  CFG=$((J % 12))
  echo "cell: hopse sweep dataset=$DATASET config=$CFG (index $IDX)"
  python "$REPO/experiments/run_hopse_sweep.py" \
      --dataset "$DATASET" --config-index "$CFG"
fi
