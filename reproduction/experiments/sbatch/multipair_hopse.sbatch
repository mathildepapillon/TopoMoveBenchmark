#!/bin/bash
#SBATCH --job-name=toposhap-multipair-hopse
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-s%a-%j.out
#SBATCH --array=42-44
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
python "$REPO/experiments/mantra_multipair_hopse.py" "$SLURM_ARRAY_TASK_ID"
