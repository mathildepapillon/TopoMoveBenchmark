#!/bin/bash
# One-run recipe, single GPU. Submit:
#   sbatch experiments/slurm/recipe.sbatch <dataset> <seed> <k> <selector>
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition (default is correct); no
# custom CPU/memory requests on GPU jobs (per-GPU defaults apply).
#SBATCH --job-name=toposhap-recipe
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed

DATASET=${1:?dataset}
SEED=${2:-42}
K=${3:-6}
SELECTOR=${4:-anchored}

python "$REPO/experiments/run_recipe.py" \
    --dataset "$DATASET" --seed "$SEED" --k "$K" --selector "$SELECTOR"
