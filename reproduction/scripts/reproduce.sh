#!/bin/bash
# Regenerate every derived artifact from data/frozen/: tests, tables and
# paper macros, both figures, and the deck-number consistency checks.
# PATH or at $TECTONIC).
#
# Frozen inputs are ingested per data/INGEST.md (never recomputed here);
# training / attribution compute lives in experiments/slurm/ and is NOT
# rerun by this script. The chain this script guarantees:
#
#   data/frozen/*  -->  paper/results_macros.tex + paper/tables/*
#                  -->  figures/lead_figure.pdf (headline variant)
#                  -->  figures/mantra_attr_pair.png
#                  -->  deck numbers verified against the same parquets
#
# Usage: scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "== 1/5 tests =="
python -m pytest -q

echo "== 2/5 tables + paper macros (data/frozen -> paper/) =="
python scripts/make_tables.py --all
python scripts/make_paper_tables.py       # t1_graph, t2_lifted, t3_nogt

echo "== 3/5 lead figures (section 6 canonical + appendix variants) =="
python figures/render_lead_figures.py                # lead_xai + lead_bench
python figures/make_overview_figure.py               # Figure 1 (overview.pdf)
python figures/build_lead_figure.py                  # -> lead_figure_full.pdf
python figures/build_lead_figure.py --x wall_seconds # -> lead_figure_wall.pdf

echo "== 4/5 attribution + appendix figures =="
python figures/build_mantra_attr_figure.py                   # deck PNG
python figures/build_mantra_attributions_paper.py            # -> mantra_attributions.pdf
python figures/build_chemistry_cases.py                      # -> chemistry_case_studies.pdf
python figures/build_estimator_variance.py                   # -> estimator_variance.pdf
python figures/build_exhaustive_landscape.py                 # -> exhaustive_landscape.pdf
python figures/build_seed_traces.py                          # -> seed_traces.pdf
python figures/build_substitute_commitment.py                # -> substitute_commitment.pdf
python figures/slides_assets/build_graphxai_comparison.py    # deck asset
python figures/slides_assets/make_mantra_slides.py           # deck assets

echo "== 5/5 deck number consistency =="
python scripts/check_deck_numbers.py

echo
echo "reproduce: all steps completed"
