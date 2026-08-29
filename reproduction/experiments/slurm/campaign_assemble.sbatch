#!/bin/bash
# Final CPU assembly job: freeze whatever cells exist into data/frozen
# (assemble_mantra + assemble_hopse_k3, both --allow-missing) and write
# results/CAMPAIGN_DONE with a completed/missing inventory. Exits 0 even on
# partial — the marker's status line distinguishes COMPLETE from PARTIAL.
#   sbatch [--dependency=afterany:JOB] experiments/slurm/campaign_assemble.sbatch
#
# Cluster policy (org-managed): no --partition. CPU-only (no --gres).
#SBATCH --job-name=toposhap-campaign-assemble
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -uo pipefail
REPO=REPO_ROOT
mkdir -p "$REPO/slurm_logs"
source "$REPO/.venv/bin/activate"
python "$REPO/experiments/assemble_campaign.py"
