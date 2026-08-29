#!/bin/bash
#SBATCH --job-name=toposhap-graph-baselines
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-23%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
# cheap methods only; SubgraphX runs in graph_baselines_sgx.sbatch
DATASETS=(benzene fluoride_carbonyl mutagenicity_gxai alkane_carbonyl)
ARCHS=(gcn gin)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
DS=${DATASETS[$((i / 6))]}
ARCH=${ARCHS[$(((i / 3) % 2))]}
SEED=${SEEDS[$((i % 3))]}
python "$REPO/experiments/run_graph_baselines.py" "$DS" "$ARCH" "$SEED" \
  random,gnnexplainer,pgexplainer
