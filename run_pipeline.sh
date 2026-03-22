#!/bin/bash
# run_pipeline.sh
# ───────────────
# Activates the "exoplanet" venv and runs the full exoplanet detection pipeline.
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   -m, --max-targets N     Max number of targets to fetch/train on (default: 300)
#   -t, --toi-only          Skip BLS during training (fast mode)
#   -s, --skip-fetch        Skip fetch step if labeled_tess_dataset.csv already exists
#   -T, --test-only         Only run test.py (skip fetch and train)
#   -h, --help              Show this message

set -e  # exit on any error

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_TARGETS=300
TOI_ONLY=false
SKIP_FETCH=false
TEST_ONLY=false
VENV_NAME="exoplanet"
CSV_FILE="labeled_tess_dataset.csv"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--max-targets)
            MAX_TARGETS="$2"
            shift 2
            ;;
        -t|--toi-only)
            TOI_ONLY=true
            shift
            ;;
        -s|--skip-fetch)
            SKIP_FETCH=true
            shift
            ;;
        -T|--test-only)
            TEST_ONLY=true
            shift
            ;;
        -h|--help)
            head -n 20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Use -h for help."
            exit 1
            ;;
    esac
done

# ── Locate and activate venv ──────────────────────────────────────────────────
# Search common locations: current dir, home, and script's own directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH=""

for candidate in \
    "$SCRIPT_DIR/$VENV_NAME" \
    "$HOME/$VENV_NAME" \
    "./$VENV_NAME"; do
    if [[ -f "$candidate/bin/activate" ]]; then
        VENV_PATH="$candidate"
        break
    fi
done

if [[ -z "$VENV_PATH" ]]; then
    echo -e "${RED}ERROR: Could not find venv '$VENV_NAME'.${NC}"
    echo "Expected it at one of:"
    echo "  $SCRIPT_DIR/$VENV_NAME"
    echo "  $HOME/$VENV_NAME"
    echo "  ./$VENV_NAME"
    echo ""
    echo "Create it with:  python3 -m venv $VENV_NAME"
    exit 1
fi

echo -e "${CYAN}Activating venv: $VENV_PATH${NC}"
source "$VENV_PATH/bin/activate"

# ── Print run config ──────────────────────────────────────────────────────────
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  ExoPlanet Finder Pipeline${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Max targets : ${YELLOW}$MAX_TARGETS${NC}"
echo -e "  TOI only    : ${YELLOW}$TOI_ONLY${NC}"
echo -e "  Skip fetch  : ${YELLOW}$SKIP_FETCH${NC}"
echo -e "  Test only   : ${YELLOW}$TEST_ONLY${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Step 1: Fetch dataset ─────────────────────────────────────────────────────
if [[ "$TEST_ONLY" == false ]]; then
    if [[ "$SKIP_FETCH" == true && -f "$CSV_FILE" ]]; then
        echo -e "${YELLOW}[Step 1] Skipping fetch — $CSV_FILE already exists.${NC}"
    else
        echo -e "${GREEN}[Step 1] Fetching labeled TESS dataset...${NC}"
        python fetch_tess_labeled_dataset.py --max-targets "$MAX_TARGETS" --output "$CSV_FILE"
        echo -e "${GREEN}[Step 1] Done. Saved to $CSV_FILE${NC}"
    fi
    echo ""

    # ── Step 2: Train classifier ──────────────────────────────────────────────
    echo -e "${GREEN}[Step 2] Training classifier...${NC}"
    TRAIN_ARGS="--csv $CSV_FILE"
    if [[ "$TOI_ONLY" == true ]]; then
        TRAIN_ARGS="$TRAIN_ARGS --toi-only"
        echo -e "  ${YELLOW}(TOI-only mode — BLS skipped)${NC}"
    fi
    python train_classifier.py $TRAIN_ARGS
    echo -e "${GREEN}[Step 2] Done. Model saved to exoplanet_rf_classifier.joblib${NC}"
    echo ""
fi

# ── Step 3: Test ──────────────────────────────────────────────────────────────
echo -e "${GREEN}[Step 3] Running test...${NC}"
python test.py
echo -e "${GREEN}[Step 3] Done.${NC}"
