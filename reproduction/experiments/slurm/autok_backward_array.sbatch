#!/bin/bash
# Auto-k BACKWARD-elimination arm: 35 (dataset, seed) cells, one GPU each.
#   sbatch experiments/slurm/autok_backward_array.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition (default is correct); no
# custom CPU/memory requests on GPU jobs (per-GPU defaults apply).
# Throttle %8; 90-min limit enforces the per-cell budget cap.
#SBATCH --job-name=toposhap-autok-bwd
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-34%8
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai NCI1 \
          cocitation_cora cocitation_citeseer cocitation_pubmed)
SEEDS=(42 43 44 45 46)
IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$((IDX / 5))]}
SEED=${SEEDS[$((IDX % 5))]}

echo "cell: dataset=$DATASET seed=$SEED (array index $IDX, backward)"
python "$REPO/experiments/run_autok.py" \
    --dataset "$DATASET" --seed "$SEED" --direction backward
