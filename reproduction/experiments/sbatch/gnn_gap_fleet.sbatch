#!/bin/bash
#SBATCH --job-name=toposhap-gnn-gaps
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-47%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
# Part-1 gap fleets on the baseline GNNs:
#  0-23: output 1 (Shapley attribution), participation semantics
# 24-47: output 2 (coalition), ZERO-FILL semantics (the ablation that
#        tests whether the fluoride loss / GCN flip is semantics choice)
DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
ARCHS=(gcn gin)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 24 ]; then METHOD=shapley; SEM=participation; j=$i
else METHOD=select; SEM=zerofill; j=$((i - 24)); fi
DS=${DATASETS[$((j / 6))]}
ARCH=${ARCHS[$(((j / 3) % 2))]}
SEED=${SEEDS[$((j % 3))]}
python "$REPO/experiments/select_on_gnn.py" "$DS" "$ARCH" "$SEED" "$METHOD" "$SEM"
