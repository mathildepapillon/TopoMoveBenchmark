#!/bin/bash
#SBATCH --job-name=toposhap-select-shard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-t%a-%j.out
#SBATCH --array=0-83%16
set -euo pipefail
REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GEA_LIMIT=0
# TopoSHAP-Select at full population on the k6 and k11 subjects.
# Within each 42-task arm: 0-23 benzene (3 seeds x 8 shards), 24-35
# mutagenicity (3 x 4), 36-38 fluoride, 39-41 alkane.
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 42 ]; then KARM=k6; j=$i; else KARM=k11; j=$((i - 42)); fi
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
