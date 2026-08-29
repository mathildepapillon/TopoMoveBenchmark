#!/bin/bash
# Run this ON the the original compute cluster (or paste it to the old thread's agent —
# thread th_01kzs75nj4eksa84n7w25vv3fr). It bundles exactly the TopoSHAP
# result inventory (data/INGEST.md) into one tarball you can move anywhere.
#
#   bash make_ingest_bundle.sh [OUTPUT.tar.gz]
#
# Then on local-cluster:
#   mkdir -p data/_staging && tar -xzf toposhap_ingest_*.tar.gz -C data/_staging
#   python scripts/ingest_results.py --staging data/_staging --verify

set -uo pipefail

OUT=${1:-toposhap_ingest_$(date -u +%Y%m%dT%H%M%SZ).tar.gz}
EXP=/srv/silico-state/_shared/silico/experiments
THREAD=/srv/silico-state/_shared/silico/threads/th_01kzs75nj4eksa84n7w25vv3fr
UPLOADS=/srv/silico-state/users/u-d7c40246f798/.silico/lab-uploads/th_01kzs75nj4eksa84n7w25vv3fr
FLAT=/mnt/delicate-frog/artifacts/silico/experiments/_flat

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
missing=0

grab() {  # grab <src> <dest-subdir>
    local src=$1 dest=$2
    if [ -e "$src" ]; then
        mkdir -p "$STAGE/$dest"
        cp -r "$src" "$STAGE/$dest/" && echo "  ok  $src"
    else
        echo "  MISSING  $src"
        missing=$((missing + 1))
    fi
}

echo "== experiment #12 results =="
R12=$EXP/[record-id]/worktree/experiments/experiment-12-rbzwyh/results
for f in analysis12 cost_wallclock12 menu12 cheap_sim12 menu_extraction12 smokegate12; do
    grab "$R12/$f.json" exp12
done

echo "== experiment #9 landscape =="
grab "$EXP/exp_01kzwzpffcfy4t7s1szphdv1yt/worktree/experiments/experiment-9-hdv1yt/results/pooled.json" exp9

echo "== experiment #11 warm-vs-cold =="
grab "$EXP/[record-id]/worktree/experiments/experiment-11-6j7m02/results/rich/cold_compare11.json" exp11

echo "== in-flight artifact roots (#13 HOPSE, #14 marker arm) =="
# EIDs matched by unique prefix; cluster-side under delicate-frog
for d in "$FLAT"/[record-id]* "$FLAT"/[record-id]*; do
    [ -e "$d" ] && grab "$d" flat || true
done
ls -d "$FLAT"/[record-id]* >/dev/null 2>&1 || { echo "  MISSING  $FLAT/[record-id]*"; missing=$((missing+1)); }
ls -d "$FLAT"/[record-id]* >/dev/null 2>&1 || { echo "  MISSING  $FLAT/[record-id]*"; missing=$((missing+1)); }

echo "== figure sources + uploads =="
grab "$THREAD/figure-sources/fig-a09699c2f95b" figure-sources
grab "$UPLOADS" lab-uploads

echo "== validated original explain module + synthesis report =="
grab "$EXP/exp_01kzs88s6dee39623jnkymp9z4/worktree" exp1_worktree
grab "$EXP/exp_01kzzch65tfvvtz3z4y36rmj01/worktree" exp10_synthesis

echo
# exclude heavyweight run artifacts — we want code + result JSONs/parquets,
# not checkpoints or raw datasets
tar -czf "$OUT" -C "$STAGE" \
    --exclude='*.ckpt' --exclude='*.pt' --exclude='*.pth' \
    --exclude='.git' --exclude='wandb' --exclude='datasets' \
    --exclude='__pycache__' .
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)); $missing item(s) missing"
[ "$missing" -eq 0 ] || echo "NOTE: missing items listed above — bundle is partial"
