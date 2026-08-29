#!/bin/bash
#SBATCH --job-name=toposhap-select-lc
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-125%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
export TOPOSHAP_LIFT_CONSISTENT=1
# Explanation output under LIFT-CONSISTENT semantics (feature-cache
# leak closed): 3 k-arms x the standard 42-task layout. Files carry
# the _lc suffix; the cached-semantics runs remain untouched.
KARMS=(k3 k6 k11)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
KARM=${KARMS[$((i / 42))]}
j=$((i % 42))
if [ "$j" -lt 24 ]; then
  DS=benzene; SEED=${SEEDS[$((j / 8))]}
  export TOPOSHAP_GEA_SHARD="$((j % 8))/8"
elif [ "$j" -lt 36 ]; then
  k=$((j - 24))
  DS=mutagenicity_gxai; SEED=${SEEDS[$((k / 4))]}
  export TOPOSHAP_GEA_SHARD="$((k % 4))/4"
else
  k=$((j - 36))
  DATASETS=(fluoride_carbonyl alkane_carbonyl)
  DS=${DATASETS[$((k / 3))]}
  SEED=${SEEDS[$((k % 3))]}
fi
python "$REPO/experiments/molecule_select_gea.py" "$DS" "$SEED" "$KARM"
