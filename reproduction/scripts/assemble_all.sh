#!/bin/bash
# Rebuild every frozen parquet from the ingested staging tree + campaign
# results — the missing link between `scripts/ingest_results.py` and
# `scripts/reproduce.sh` on a fresh clone.
#
#   scripts/ingest_results.py --staging …   # unpack data/_bundle into place
#   scripts/assemble_all.sh                 # <- this script
#   scripts/reproduce.sh                    # tables, figures, checks
#
# Assemblers are deterministic: on an already-assembled tree they rewrite
# byte-identical files, so running this after the results freeze is safe
# (verify with `python scripts/freeze_results.py --check` afterwards; note
# the check re-hashes ~190k files and takes a few minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python experiments/assemble_autok.py
python experiments/assemble_autok.py --backward
python experiments/assemble_ladder_b2.py
python experiments/assemble_ladder_b2.py --t1
python experiments/assemble_ladder_b2.py --hopse
python experiments/assemble_ladder_b2.py --hopse --t1
python experiments/assemble_anchored3.py
python experiments/assemble_campaign.py
python experiments/assemble_hopse_k3.py
python experiments/assemble_hopse_sweep.py
python experiments/assemble_mantra.py
python scripts/baseline_markers.py

echo
echo "assemble_all: every frozen parquet regenerated"
