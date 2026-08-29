#!/bin/bash
#SBATCH --job-name=toposhap-gea-k11-shard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-41%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
# k=11 (full-vocabulary reference subjects) at the FULL gt-positive
# population. Same shard layout as the k3/k6 fleets: 0-23 benzene
# (3 seeds x 8 shards); 24-35 mutagenicity (3 x 4); 36-38 fluoride;
# 39-41 alkane.
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 24 ]; then
  DS=benzene; SEED=${SEEDS[$((i / 8))]}
  export TOPOSHAP_GEA_SHARD="$((i % 8))/8"
elif [ "$i" -lt 36 ]; then
  j=$((i - 24))
  DS=mutagenicity_gxai; SEED=${SEEDS[$((j / 4))]}
  export TOPOSHAP_GEA_SHARD="$((j % 4))/4"
else
  j=$((i - 36))
  DATASETS=(fluoride_carbonyl alkane_carbonyl)
  DS=${DATASETS[$((j / 3))]}
  SEED=${SEEDS[$((j % 3))]}
fi
python "$REPO/experiments/molecule_participation_gea.py" "$DS" "$SEED" 128 participation k11
