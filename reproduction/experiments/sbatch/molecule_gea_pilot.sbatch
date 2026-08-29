#!/bin/bash
#SBATCH --job-name=toposhap-gea-pilot
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-8
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
COMBOS=(
  "42 256 signal"
  "43 256 signal"
  "44 256 signal"
  "42 256 participation"
  "43 256 participation"
  "44 256 participation"
  "42 512 participation"
  "43 512 participation"
  "44 512 participation"
)
read -r SEED PASSES GAME <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
python "$REPO/experiments/molecule_participation_gea.py" benzene "$SEED" "$PASSES" "$GAME"
