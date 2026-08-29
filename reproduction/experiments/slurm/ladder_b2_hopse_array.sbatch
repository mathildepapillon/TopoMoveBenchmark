#!/bin/bash
# LADDER B=2 HOPSE arm (amended): 12 cells — 4 molecule datasets x seeds
# {42,43,44}, matching the #13 sweep baselines. One GPU each.
#   sbatch experiments/slurm/ladder_b2_hopse_array.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition (default is correct); no
# custom CPU/memory requests on GPU jobs (per-GPU defaults apply).
# Throttle %16 (researcher directive 2026-08-17); 90-min limit enforces the
# per-cell budget cap.
#SBATCH --job-name=toposhap-ladder-b2-hopse
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-11%16
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai NCI1)
SEEDS=(42 43 44)
IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

echo "cell: dataset=$DATASET seed=$SEED (array index $IDX, ladder B=2 HOPSE)"
python "$REPO/experiments/run_ladder_b2_hopse.py" \
    --dataset "$DATASET" --seed "$SEED"
