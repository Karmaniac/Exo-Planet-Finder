#!/bin/bash
# run_pipeline.sh
# ───────────────
# Activates the "exoplanet" venv and runs the exoplanet detection pipeline.
#
# SUBCOMMANDS:
#   pipeline   Run the full pipeline (fetch → train → test)  [default]
#   predict    Classify a single TIC target directly
#
# PIPELINE usage:
#   ./run_pipeline.sh [pipeline] [OPTIONS]
#
#   -m, --max-targets N     Max number of targets to fetch/train on (default: 300)
#   -t, --toi-only          Skip BLS during training (fast mode)
#   -s, --skip-fetch        Skip fetch step if labeled_tess_dataset.csv already exists
#   -T, --test-only         Only run test.py (skip fetch and train)
#   -h, --help              Show this message
#
# PREDICT usage:
#   ./run_pipeline.sh predict <TIC_ID> [OPTIONS]
#
#   -s, --sector N          Specific TESS sector to use (default: first available)
#   --no-plot               Skip the folded light curve plot
#   -h, --help              Show this message
#
# EXAMPLES:
#   ./run_pipeline.sh                              # full pipeline, 300 targets
#   ./run_pipeline.sh --max-targets 100 --toi-only # fast training run
#   ./run_pipeline.sh --skip-fetch                 # skip fetch, retrain only
#   ./run_pipeline.sh predict 307210830            # classify a TIC target
#   ./run_pipeline.sh predict 381472147 --sector 14
#   ./run_pipeline.sh predict 307210830 --no-plot

set -e

VENV_NAME="exoplanet"
CSV_FILE="labeled_tess_dataset.csv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ── Locate and activate venv ──────────────────────────────────────────────────
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

source "$VENV_PATH/bin/activate"

# ── Subcommand routing ────────────────────────────────────────────────────────
SUBCOMMAND="pipeline"   # default

if [[ $# -gt 0 && "$1" == "predict" ]]; then
    SUBCOMMAND="predict"
    shift
elif [[ $# -gt 0 && "$1" == "pipeline" ]]; then
    SUBCOMMAND="pipeline"
    shift
fi

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT subcommand
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$SUBCOMMAND" == "predict" ]]; then

    TIC_ID=""
    SECTOR=""
    NO_PLOT=false

    if [[ $# -gt 0 && "$1" != -* ]]; then
        TIC_ID="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--sector)
                SECTOR="$2"
                shift 2
                ;;
            --no-plot)
                NO_PLOT=true
                shift
                ;;
            -h|--help)
                echo "Usage: ./run_pipeline.sh predict <TIC_ID> [--sector N] [--no-plot]"
                echo ""
                echo "Examples:"
                echo "  ./run_pipeline.sh predict 307210830"
                echo "  ./run_pipeline.sh predict 381472147 --sector 14"
                echo "  ./run_pipeline.sh predict 307210830 --no-plot"
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown argument: $1${NC}"
                echo "Usage: ./run_pipeline.sh predict <TIC_ID> [--sector N] [--no-plot]"
                exit 1
                ;;
        esac
    done

    if [[ -z "$TIC_ID" ]]; then
        echo -e "${RED}ERROR: TIC ID required.${NC}"
        echo "Usage: ./run_pipeline.sh predict <TIC_ID> [--sector N] [--no-plot]"
        echo "Example: ./run_pipeline.sh predict 307210830"
        exit 1
    fi

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  ExoPlanet Finder — Predict${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  TIC ID  : ${YELLOW}$TIC_ID${NC}"
    echo -e "  Sector  : ${YELLOW}${SECTOR:-auto}${NC}"
    echo -e "  Plot    : ${YELLOW}$([ "$NO_PLOT" == true ] && echo no || echo yes)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    PREDICT_ARGS="$TIC_ID"
    [[ -n "$SECTOR"       ]] && PREDICT_ARGS="$PREDICT_ARGS --sector $SECTOR"
    [[ "$NO_PLOT" == true ]] && PREDICT_ARGS="$PREDICT_ARGS --no-plot"

    python predict.py $PREDICT_ARGS
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE subcommand (default)
# ══════════════════════════════════════════════════════════════════════════════

MAX_TARGETS=300
TOI_ONLY=false
SKIP_FETCH=false
TEST_ONLY=false

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
            grep "^#" "$0" | head -n 30 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Use -h for help."
            exit 1
            ;;
    esac
done

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  ExoPlanet Finder — Pipeline${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Max targets : ${YELLOW}$MAX_TARGETS${NC}"
echo -e "  TOI only    : ${YELLOW}$TOI_ONLY${NC}"
echo -e "  Skip fetch  : ${YELLOW}$SKIP_FETCH${NC}"
echo -e "  Test only   : ${YELLOW}$TEST_ONLY${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Step 1: Fetch ─────────────────────────────────────────────────────────────
if [[ "$TEST_ONLY" == false ]]; then
    if [[ "$SKIP_FETCH" == true && -f "$CSV_FILE" ]]; then
        echo -e "${YELLOW}[Step 1] Skipping fetch — $CSV_FILE already exists.${NC}"
    else
        echo -e "${GREEN}[Step 1] Fetching labeled TESS dataset...${NC}"
        python fetch_tess_labeled_dataset.py --max-targets "$MAX_TARGETS" --output "$CSV_FILE"
        echo -e "${GREEN}[Step 1] Done. Saved to $CSV_FILE${NC}"
    fi
    echo ""

    # ── Step 2: Train ─────────────────────────────────────────────────────────
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
