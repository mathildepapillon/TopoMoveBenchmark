#!/bin/bash
#SBATCH --job-name=toposhap-gea-k6-full
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-11%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
DS=${DATASETS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}
python "$REPO/experiments/molecule_participation_gea.py" "$DS" "$SEED" 128 participation k6
