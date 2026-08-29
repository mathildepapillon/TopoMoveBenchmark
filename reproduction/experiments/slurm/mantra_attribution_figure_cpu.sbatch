#!/bin/bash
# Per-cell TopoSHAP attributions + counting evidence on MANTRA for the
# deck figure (three seed-43 models, matched-count test pair). Read-only
# w.r.t. campaign artifacts; safe alongside the AUROC backfill.
#   sbatch [--dependency=afterany:JOB] experiments/slurm/mantra_attribution_figure.sbatch
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU
# jobs. 90-min cap.
#SBATCH --job-name=toposhap-mantra-attrfig
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
REPO=REPO_ROOT
mkdir -p "$REPO/slurm_logs"
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

python "$REPO/experiments/mantra_attribution_figure.py"
