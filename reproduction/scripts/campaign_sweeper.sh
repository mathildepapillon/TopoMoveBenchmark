#!/bin/bash
# Cluster-native self-heal for the MANTRA + k3-rung campaign. CPU-only,
# idempotent, bounded (no self-rescheduling):
#   (a) exits immediately if results/CAMPAIGN_DONE says COMPLETE;
#   (b) otherwise resubmits exactly the missing cells (the runners skip
#       completed JSONLs, so overlap with still-running jobs is harmless)
#       chained into a fresh assembly job.
# Submit as delayed retry passes, e.g.:
#   sbatch --begin=now+2hour scripts/campaign_sweeper.sh
#   sbatch --begin=now+4hour scripts/campaign_sweeper.sh
#
# Cluster policy (org-managed): no --partition; %16 array throttle.
#SBATCH --job-name=toposhap-campaign-sweeper
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%x-%j.out

set -uo pipefail
REPO=~/code/toposhap
cd "$REPO"
mkdir -p slurm_logs
source "$REPO/.venv/bin/activate"

if grep -q '^status: COMPLETE' results/CAMPAIGN_DONE 2>/dev/null; then
    echo "CAMPAIGN_DONE says COMPLETE; nothing to do"
    exit 0
fi

# Defer while the campaign's own jobs are still queued/running: the runners
# skip completed cells, but there is no point double-submitting in-flight
# work — the next sweeper pass (or the running DAG itself) will finish it.
INFLIGHT=$(squeue -h -u "$USER" \
    -n toposhap-mantra,toposhap-hopse-k3,toposhap-mantra-balanced,toposhap-campaign-assemble \
    -o %i | wc -l)
if [ "$INFLIGHT" -gt 0 ]; then
    echo "$INFLIGHT campaign job(s) still queued/running; deferring"
    exit 0
fi

mapfile -t MISSING < <(python experiments/campaign_missing.py)
MANTRA_IDX="${MISSING[0]:-}"
K3_IDX="${MISSING[1]:-}"
echo "missing mantra_arrays indices: ${MANTRA_IDX:-none}"
echo "missing hopse_k3_array indices: ${K3_IDX:-none}"

DEP=""
if [ -n "$MANTRA_IDX" ]; then
    A=$(sbatch --parsable --array="${MANTRA_IDX}%16" \
        experiments/slurm/mantra_arrays.sbatch)
    echo "resubmitted mantra_arrays as job $A (array ${MANTRA_IDX})"
    DEP="--dependency=afterany:$A"
fi
if [ -n "$K3_IDX" ]; then
    B=$(sbatch --parsable $DEP --array="${K3_IDX}%16" \
        experiments/slurm/hopse_k3_array.sbatch)
    echo "resubmitted hopse_k3_array as job $B (array ${K3_IDX})"
    DEP="--dependency=afterany:$B"
fi
# balanced-accuracy backfill (MANTRA cells): needed if any existing cell
# lacks the fields, or if any mantra cell was just resubmitted above
if ! python experiments/mantra_balanced_eval.py --check \
        || [ -n "$MANTRA_IDX" ] || [ -n "$K3_IDX" ]; then
    E=$(sbatch --parsable $DEP experiments/slurm/mantra_balanced_eval.sbatch)
    echo "balanced-eval job $E ${DEP:+(depends ${DEP})}"
    DEP="--dependency=afterany:$E"
fi
C=$(sbatch --parsable $DEP experiments/slurm/campaign_assemble.sbatch)
echo "assembly job $C ${DEP:+(depends ${DEP})}"
