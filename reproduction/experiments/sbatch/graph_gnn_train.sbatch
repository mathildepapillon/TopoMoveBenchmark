#!/bin/bash
#SBATCH --job-name=toposhap-gnn-train
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-2%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
# alkane_carbonyl already trained locally (results/graph_baselines)
DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
python "$REPO/experiments/train_graph_gnns.py" "$DS"
