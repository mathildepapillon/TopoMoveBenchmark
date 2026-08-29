#!/bin/bash
#SBATCH --job-name=toposhap-graph-sgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-35%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
# SubgraphX (~30s+/graph): benzene's 917 graphs are sharded 3x.
# 0-17 = benzene, (arch, seed, shard); 18-35 = the other three datasets.
ARCHS=(gcn gin)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 18 ]; then
  DS=benzene
  ARCH=${ARCHS[$((i / 9))]}
  SEED=${SEEDS[$(((i / 3) % 3))]}
  export TOPOSHAP_GRAPH_BASELINE_SHARD="$((i % 3))/3"
else
  j=$((i - 18))
  DATASETS=(fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
  DS=${DATASETS[$((j / 6))]}
  ARCH=${ARCHS[$(((j / 3) % 2))]}
  SEED=${SEEDS[$((j % 3))]}
fi
python "$REPO/experiments/run_graph_baselines.py" "$DS" "$ARCH" "$SEED" subgraphx
