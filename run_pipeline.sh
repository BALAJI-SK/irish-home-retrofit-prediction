#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh  —  Irish Home Retrofit Prediction Pipeline
# Usage: bash run_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMP_DIR="$SCRIPT_DIR/valid-imp"
PYTHON="$IMP_DIR/venv/bin/python"
SCRIPTS_DIR="$IMP_DIR/scripts"

# Colour helpers
GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; RESET='\033[0m'

run_step() {
    local step="$1"
    local script="$2"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════${RESET}"
    echo -e "${CYAN}  STEP $step — $(basename "$script")${RESET}"
    echo -e "${CYAN}════════════════════════════════════════════════════${RESET}"
    if "$PYTHON" "$script"; then
        echo -e "${GREEN}  ✓ Step $step complete${RESET}"
    else
        echo -e "${RED}  ✗ Step $step FAILED — pipeline aborted${RESET}"
        exit 1
    fi
}

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║   Irish Home Retrofit Prediction Pipeline        ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
START=$(date +%s)

run_step 1 "$SCRIPTS_DIR/01_clean_and_prepare.py"
run_step 2 "$SCRIPTS_DIR/02_county_profile.py"
run_step 3 "$SCRIPTS_DIR/03_train_model.py"
run_step 4 "$SCRIPTS_DIR/04_equity_gap.py"
run_step 5 "$SCRIPTS_DIR/05_xai_explainer.py"

END=$(date +%s)
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Pipeline complete in $((END - START))s${RESET}"
echo -e "${GREEN}  Outputs written to: $IMP_DIR/outputs/${RESET}"
echo -e "${GREEN}════════════════════════════════════════════════════${RESET}"
echo ""
