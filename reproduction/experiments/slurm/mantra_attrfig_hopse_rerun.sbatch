#!/bin/bash
# HOPSE pair attributions at the encoding interface (fixes the degenerate
# CellMaskingGame result in job 419349's attr_fig_data.json — HOPSE never
# reads x_{rank}). Pair games only; menu/GCCN results stay as recorded.
#   sbatch experiments/slurm/mantra_attrfig_hopse_rerun.sbatch
#
# Cluster policy (org-managed): no --partition; CPU job, so explicit CPU
# request is allowed.
#SBATCH --job-name=toposhap-hopse-attr-rerun
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
REPO=REPO_ROOT
mkdir -p "$REPO/slurm_logs"
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed

python "$REPO/experiments/mantra_attrfig_hopse_rerun.py"
