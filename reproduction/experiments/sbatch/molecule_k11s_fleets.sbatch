#!/bin/bash
#SBATCH --job-name=toposhap-k11s-gea
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-83%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
# Both outputs on the STRUCTURAL-lift subjects (k11s): constant
# higher-rank features, so no feature cache exists and the standard
# participation game is already faithful. 0-41 explanation (cheap),
# 42-83 attribution (128 passes). Standard 42-task layout inside each.
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 42 ]; then MODE=select; j=$i; else MODE=attr; j=$((i - 42)); fi
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
if [ "$MODE" = select ]; then
  python "$REPO/experiments/molecule_select_gea.py" "$DS" "$SEED" k11s
else
  python "$REPO/experiments/molecule_participation_gea.py" "$DS" "$SEED" 128 participation k11s
fi
