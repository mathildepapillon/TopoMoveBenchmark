#!/bin/bash
# Tier-1 GCCN arms: anchored k=3 + ladder B=2 over the 3 new datasets x
# seeds {42,43,44}. 18 cells: indices 0-8 anchored, 9-17 ladder. Submit:
#   sbatch experiments/slurm/t1_arrays.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. Throttle %16 (researcher directive); 90-min per-cell cap.
#SBATCH --job-name=toposhap-t1-gccn
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-17%16
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

DATASETS=(MUTAG NCI109 alkane_carbonyl)
SEEDS=(42 43 44)
IDX=${SLURM_ARRAY_TASK_ID}
ARM=$((IDX / 9))     # 0 = anchored k=3, 1 = ladder B=2
CELL=$((IDX % 9))
DATASET=${DATASETS[$((CELL / 3))]}
SEED=${SEEDS[$((CELL % 3))]}

if [ "$ARM" -eq 0 ]; then
  echo "cell: anchored3 dataset=$DATASET seed=$SEED (index $IDX)"
  python "$REPO/experiments/run_anchored3.py" --dataset "$DATASET" --seed "$SEED"
else
  echo "cell: ladder_b2 dataset=$DATASET seed=$SEED (index $IDX)"
  python "$REPO/experiments/run_ladder_b2.py" --dataset "$DATASET" --seed "$SEED"
fi
