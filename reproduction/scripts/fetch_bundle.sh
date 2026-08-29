#!/bin/bash
# Fetch the the original compute cluster transfer bundle (stamp 20260816T153327Z) from presigned
# HTTPS links, verify every sha256, reassemble the split flat tarball, extract
# into data/_staging, and run schema-verified ingestion.
#
#   bash scripts/fetch_bundle.sh POD_URL MANIFEST_GZ_URL SPLIT_AA_URL SPLIT_AB_URL
#
# Links expire after 1 hour — run promptly after minting.

set -euo pipefail

POD_URL=${1:?pod tarball url}
MANIFEST_URL=${2:?manifest.json.gz url}
AA_URL=${3:?split-aa url}
AB_URL=${4:?split-ab url}

REPO=$(cd "$(dirname "$0")/.." && pwd)
DL=$REPO/data/_bundle
STAGING=$REPO/data/_staging
mkdir -p "$DL" "$STAGING"
cd "$DL"

STAMP=20260816T153327Z
POD=toposhap_pod_$STAMP.tar.gz
FLAT=toposhap_flat_$STAMP.tar.gz

# expected sha256s from the worker's job report (README table)
POD_SHA=9b9fd61997f4644aecc8a428e6bc034aebc62223d04759b5bde908157720bdff
MANIFEST_SHA=395099e00a208ad762c801c693256c637dc08dc1b32f597057f867ae02a093e3
AA_SHA=68fb20b3246b731eb5631fa631e754ca1fed6e8958bba3dc8340f8e2ec167d97
FLAT_SHA=792e2d20b85aec78530543c55fc981aeb270fb2868538109b4c395ab3e513999

fetch() {  # fetch <url> <outfile>
    echo "fetching $2 ..."
    curl -fSL --retry 3 -o "$2" "$1"
}

check() {  # check <file> <sha256>
    echo "$2  $1" | sha256sum -c - || { echo "SHA MISMATCH: $1"; exit 1; }
}

fetch "$POD_URL" "$POD";            check "$POD" "$POD_SHA"
fetch "$MANIFEST_URL" MANIFEST.json.gz; check MANIFEST.json.gz "$MANIFEST_SHA"
fetch "$AA_URL" "$FLAT.split-aa";   check "$FLAT.split-aa" "$AA_SHA"
fetch "$AB_URL" "$FLAT.split-ab"   # per-part sha for ab not published; whole-file check below

cat "$FLAT.split-aa" "$FLAT.split-ab" > "$FLAT"
check "$FLAT" "$FLAT_SHA"
gunzip -kf MANIFEST.json.gz

echo "extracting into $STAGING ..."
tar -xzf "$POD" -C "$STAGING"
tar -xzf "$FLAT" -C "$STAGING"
cp MANIFEST.json "$STAGING/BUNDLE_MANIFEST.json"

echo "ingesting ..."
python "$REPO/scripts/ingest_results.py" --staging "$STAGING"
python "$REPO/scripts/ingest_results.py" --verify
echo
echo "done — splits and tarballs kept in $DL (delete after freeze if space matters)"
