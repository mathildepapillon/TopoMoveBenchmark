#!/bin/bash
# Landscape sweep as a SLURM array over subset masks — BACKGROUND priority
# (oracle garnish; first thing cut if compute tightens). Submit e.g.:
#   sbatch --array=1-2047%48 experiments/slurm/landscape_array.sbatch MUTAG 42
# The %48 throttle respects the 48-GPU concurrency cap.
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-landscape
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed

DATASET=${1:?dataset}
SEED=${2:-42}
MASK=${SLURM_ARRAY_TASK_ID:?run as a SLURM array}

python "$REPO/experiments/run_landscape.py" \
    --dataset "$DATASET" --seeds "$SEED" --mask "$MASK"
