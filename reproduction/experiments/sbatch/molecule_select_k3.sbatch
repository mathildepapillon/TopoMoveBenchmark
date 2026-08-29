#!/bin/bash
#SBATCH --job-name=toposhap-select-k3
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-41%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
# Coalition output on the k=3 (anchored recipe) subjects — completes
# the alignment-ceiling-vs-k diagnostic (k3/k6/k11). Same layout as
# the k6/k11 arms: 0-23 benzene (3 seeds x 8 shards), 24-35
# mutagenicity (3 x 4), 36-38 fluoride, 39-41 alkane.
SEEDS=(42 43 44)
j=$SLURM_ARRAY_TASK_ID
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
python "$REPO/experiments/molecule_select_gea.py" "$DS" "$SEED" k3
