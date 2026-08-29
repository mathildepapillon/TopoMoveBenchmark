#!/bin/bash
# Balanced-accuracy backfill for all completed MANTRA cells (eval-only
# checkpoint reloads) + class-balance record. Idempotent and resumable —
# JSONLs are patched as cells are evaluated; safe to resubmit.
#   sbatch [--dependency=afterany:JOB] experiments/slurm/mantra_balanced_eval.sbatch
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. 90-min cap.
#SBATCH --job-name=toposhap-mantra-balanced
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
REPO=REPO_ROOT
mkdir -p "$REPO/slurm_logs"
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

python "$REPO/experiments/mantra_balanced_eval.py"
