#!/usr/bin/env bash
# Replication script for all prevalence estimation analyses.
#
# Step 0 regenerates the 4 per-analysis CSVs in estimation/estimation_inputs/
# from the master dataset (analysis/corpus_monthly.csv), then runs the 4
# Bayesian models. Posterior summaries are written to analysis/results/prevalence/.
#
# Usage:
#   bash estimation/estimate.sh

set -euo pipefail

PY=.venv/bin/python
SCRIPT=estimation/estimate_prevalence_binomial.py
INPUT_DIR=estimation/estimation_inputs

echo "============================================================"
echo "Step 0: Derive per-analysis inputs from the master dataset"
echo "============================================================"
$PY estimation/build_estimation_inputs.py

echo ""
echo "============================================================"
echo "Analysis 1: Yearly trend by LC region (HECC | CC)"
echo "============================================================"
$PY $SCRIPT \
  --input "$INPUT_DIR/prev_hecc_cc_region_yearly.csv" \
  --category health_effects_of_climate_change \
  --group-by region --time yearly --no-group-time \
  --samples 4000 --tune 3000 --chains 4 --target-accept 0.99

echo ""
echo "============================================================"
echo "Analysis 2: Monthly seasonal by climate zone (HECC | CC)"
echo "============================================================"
$PY $SCRIPT \
  --input "$INPUT_DIR/prev_hecc_cc_climate_zone_monthly.csv" \
  --category health_effects_of_climate_change \
  --group-by climate_zone --time monthly --no-global-time \
  --samples 4000 --tune 3000 --chains 4 --target-accept 0.99

echo ""
echo "============================================================"
echo "Analysis 3: HDI 2025 group association (HECC | CC)"
echo "============================================================"
$PY $SCRIPT \
  --input "$INPUT_DIR/prev_hecc_cc_hdi_category.csv" \
  --category health_effects_of_climate_change \
  --group-by hdi_category --independent-groups \
  --samples 4000 --tune 3000 --chains 4 --target-accept 0.99

echo ""
echo "============================================================"
echo "Analysis 4: Yearly trend by LC region (Health | CC)"
echo "============================================================"
$PY $SCRIPT \
  --input "$INPUT_DIR/prev_health_cc_region_yearly.csv" \
  --category health \
  --group-by region --time yearly --no-group-time \
  --samples 4000 --tune 3000 --chains 4 --target-accept 0.99

echo ""
echo "============================================================"
echo "All analyses complete. Results in analysis/results/prevalence/"
echo "============================================================"
