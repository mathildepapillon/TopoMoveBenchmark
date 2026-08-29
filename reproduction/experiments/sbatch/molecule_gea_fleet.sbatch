#!/bin/bash
#SBATCH --job-name=toposhap-gea-fleet
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-17%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
DATASETS=(fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
SEEDS=(42 43 44)
GAMES=(participation signal)
i=$SLURM_ARRAY_TASK_ID
DS=${DATASETS[$((i / 6))]}
SEED=${SEEDS[$(((i % 6) / 2))]}
GAME=${GAMES[$((i % 2))]}
python "$REPO/experiments/molecule_participation_gea.py" "$DS" "$SEED" 256 "$GAME"
