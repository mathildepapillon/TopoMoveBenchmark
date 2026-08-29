#!/bin/bash
#SBATCH --job-name=toposhap-hopse-participation-cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
python "$REPO/experiments/mantra_hopse_participation_pair.py"
