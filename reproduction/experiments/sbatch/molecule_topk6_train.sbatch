#!/bin/bash
#SBATCH --job-name=toposhap-topk6-train
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-3
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
python "$REPO/experiments/train_molecule_topk6.py" "${DATASETS[$SLURM_ARRAY_TASK_ID]}"
