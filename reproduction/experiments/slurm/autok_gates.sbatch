#!/bin/bash
# Calibration gates B + C for the auto-k arm, one GPU job. Submit:
#   sbatch experiments/slurm/autok_gates.sbatch
# Gate B: full (cocitation_cora, seed 42) auto-k pipeline; sanity of v_full,
#         k, and test accuracy vs the #14 anchored k=3 record (0.8579).
# Gate C: fresh benzene seed-45 stem vs the #14 cached game at mask 2047.
#
# Cluster policy (org-managed): no --partition; no custom CPU/mem on GPU jobs.
#SBATCH --job-name=toposhap-autok-gates
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

REPO=REPO_ROOT
source "$REPO/.venv/bin/activate"
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA="$REPO/external/TopoBench/datasets/graphxai"

echo "=== gate B: cocitation_cora seed 42, full auto-k pipeline ==="
python "$REPO/experiments/run_autok.py" --dataset cocitation_cora --seed 42

echo "=== gate C: benzene stem vs #14 cached game ==="
python "$REPO/experiments/autok_gate_c.py"
