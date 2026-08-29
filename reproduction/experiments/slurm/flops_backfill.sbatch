#!/bin/bash
# GCCN ladder-B=2 exact-FLOPs backfill: one GPU job profiling every distinct
# rung mask (seeds 42-44) plus 2047, per dataset. Submit:
#   sbatch experiments/slurm/flops_backfill.sbatch
# Wait on jobs with: monitor_jobs JOB1 [JOB2 ...]   (never sleep/sacct loops)
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-flops-backfill
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

python "$REPO/experiments/profile_ladder_masks.py"
