#!/bin/bash
#SBATCH --job-name=toposhap-multipair-gccn
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-s%a-%j.out
#SBATCH --array=42-44
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
python "$REPO/experiments/mantra_multipair_participation.py" "$SLURM_ARRAY_TASK_ID"
