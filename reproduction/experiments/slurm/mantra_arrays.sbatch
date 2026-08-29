#!/bin/bash
# MANTRA arms, one array: indices 0-2 GCCN ladder (seeds 42-44), 3-5 anchored
# k=3, 6-8 HOPSE ladder, 9-20 HOPSE sweep configs 0-11, 21-26 GCCN menu masks.
#   sbatch experiments/slurm/mantra_arrays.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. Throttle %16 (researcher directive); 90-min per-cell cap.
#SBATCH --job-name=toposhap-mantra
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-26%16
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

SEEDS=(42 43 44)
MENU=(21 26 72 129 329 479)
IDX=${SLURM_ARRAY_TASK_ID}
DS=mantra_orientation

if [ "$IDX" -lt 3 ]; then
  echo "cell: gccn ladder seed=${SEEDS[$IDX]}"
  python "$REPO/experiments/run_ladder_b2.py" --dataset $DS --seed "${SEEDS[$IDX]}"
elif [ "$IDX" -lt 6 ]; then
  echo "cell: anchored3 seed=${SEEDS[$((IDX-3))]}"
  python "$REPO/experiments/run_anchored3.py" --dataset $DS --seed "${SEEDS[$((IDX-3))]}"
elif [ "$IDX" -lt 9 ]; then
  echo "cell: hopse ladder seed=${SEEDS[$((IDX-6))]}"
  python "$REPO/experiments/run_ladder_b2_hopse.py" --dataset $DS --seed "${SEEDS[$((IDX-6))]}"
elif [ "$IDX" -lt 21 ]; then
  CFG=$((IDX-9))
  echo "cell: hopse sweep config=$CFG"
  python "$REPO/experiments/run_hopse_sweep.py" --dataset $DS --config-index "$CFG"
else
  MASK=${MENU[$((IDX-21))]}
  echo "cell: gccn menu mask=$MASK"
  python "$REPO/experiments/run_gccn_menu.py" --dataset $DS --mask "$MASK"
fi
