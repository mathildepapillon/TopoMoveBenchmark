#!/bin/bash
#SBATCH --job-name=toposhap-select-gnn
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-29%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
# Select readout on the baseline GNNs (special-case closure).
# 0-17: (benzene, fluoride, alkane) x (gcn, gin) x 3 seeds.
# 18-29: mutagenicity (200-node tail) x (gcn, gin) x 3 seeds x 2 shards.
ARCHS=(gcn gin)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 18 ]; then
  DATASETS=(benzene fluoride_carbonyl alkane_carbonyl)
  DS=${DATASETS[$((i / 6))]}
  ARCH=${ARCHS[$(((i / 3) % 2))]}
  SEED=${SEEDS[$((i % 3))]}
else
  j=$((i - 18))
  DS=mutagenicity_gxai
  ARCH=${ARCHS[$((j / 6))]}
  SEED=${SEEDS[$(((j / 2) % 3))]}
  export TOPOSHAP_GRAPH_BASELINE_SHARD="$((j % 2))/2"
fi
python "$REPO/experiments/select_on_gnn.py" "$DS" "$ARCH" "$SEED"
