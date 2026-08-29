#!/bin/bash
#SBATCH --job-name=toposhap-gea-k6-shard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-35%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
# benzene (917 graphs) and mutagenicity (200-player tail) exceed the 12h
# window as single jobs: shard them. 0-23 = benzene, 3 seeds x 8 shards;
# 24-35 = mutagenicity, 3 seeds x 4 shards.
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 24 ]; then
  DS=benzene; SEED=${SEEDS[$((i / 8))]}
  export TOPOSHAP_GEA_SHARD="$((i % 8))/8"
else
  j=$((i - 24))
  DS=mutagenicity_gxai; SEED=${SEEDS[$((j / 4))]}
  export TOPOSHAP_GEA_SHARD="$((j % 4))/4"
fi
python "$REPO/experiments/molecule_participation_gea.py" "$DS" "$SEED" 128 participation k6
