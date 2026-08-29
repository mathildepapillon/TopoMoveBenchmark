#!/bin/bash
# Anchored k=3 measurement-exact arm: 21 cells — 7 datasets x seeds
# {42,43,44}. One GPU each. Submit:
#   sbatch experiments/slurm/anchored3_array.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition (default is correct); no
# custom CPU/memory requests on GPU jobs (per-GPU defaults apply).
# Throttle %16 (researcher directive 2026-08-17); 90-min limit enforces the
# per-cell budget cap.
#SBATCH --job-name=toposhap-anchored3
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

DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai NCI1 \
          cocitation_cora cocitation_citeseer cocitation_pubmed)
SEEDS=(42 43 44)
IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

echo "cell: dataset=$DATASET seed=$SEED (array index $IDX, anchored k=3)"
python "$REPO/experiments/run_anchored3.py" --dataset "$DATASET" --seed "$SEED"
