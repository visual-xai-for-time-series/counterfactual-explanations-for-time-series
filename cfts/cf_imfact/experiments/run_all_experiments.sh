#!/usr/bin/env bash
# Run all IMFACT experiments: ablations and method comparisons for both datasets.
#
# Sequentially executes:
#   1. faultdetectiona/ablation_faultdetectiona.py
#   2. faultdetectiona/compare_faultdetectiona.py
#   3. fruitflies/ablation_fruitflies.py
#   4. fruitflies/compare_fruitflies.py
#
# Usage:
#   ./run_all_experiments.sh [options]
#
#   ./run_all_experiments.sh --out-dir ./results --n-samples 20 --ablation-max-samples 8
#   ./run_all_experiments.sh --skip faultdetectiona_ablation fruitflies_ablation
#
# Options:
#   --out-dir DIR                Root output directory (default: ./results)
#   --n-samples N                Test samples for compare scripts (default: 50)
#   --fruitflies-n-samples N     Override --n-samples for FruitFlies compare (default: same as --n-samples)
#   --ablation-max-samples N     --max-samples for ablation scripts (default: 8)
#   --ablation-max-plot-samples N  --max-plot-samples for ablation scripts (default: 1)
#   --multi-nun-counts COUNTS    Comma-separated n_nuns values (default: 2,3,5)
#   --downsample N               Downsample factor for ablation scripts (default: 1)
#   --seed N                     Random seed for compare scripts (default: 42)
#   --skip NAMES...              Space-separated experiment names to skip
#   --only NAMES...              Run only these experiments (overrides --skip)
#   --exclude-glacier-long       Exclude Glacier from long-series experiments (FaultDetectionA, FruitFlies)
#   --exclude-mascots-long       Exclude MASCOTS from long-series experiments (FaultDetectionA, FruitFlies) (default: on)
#   --include-mascots-long       Include MASCOTS in long-series experiments (overrides the default exclusion)
#   --help                       Show this help and exit
#
# Available experiment names: faultdetectiona_ablation faultdetectiona_compare
#                              fruitflies_ablation fruitflies_compare

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OUT_DIR="$SCRIPT_DIR/results"
N_SAMPLES=50
FRUITFLIES_N_SAMPLES=""
ABLATION_MAX_SAMPLES=8
ABLATION_MAX_PLOT_SAMPLES=1
MULTI_NUN_COUNTS="2,3,5"
DOWNSAMPLE=1
SEED=42
SKIP=()
ONLY=()
INCLUDE_GLACIER_LONG=1
INCLUDE_MASCOTS_LONG=0

ALL_EXPERIMENTS=(
    faultdetectiona_ablation
    faultdetectiona_compare
    fruitflies_ablation
    fruitflies_compare
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)                   OUT_DIR="$2";                    shift 2 ;;
        --n-samples)                 N_SAMPLES="$2";                  shift 2 ;;
        --fruitflies-n-samples)      FRUITFLIES_N_SAMPLES="$2";       shift 2 ;;
        --ablation-max-samples)      ABLATION_MAX_SAMPLES="$2";       shift 2 ;;
        --ablation-max-plot-samples) ABLATION_MAX_PLOT_SAMPLES="$2";  shift 2 ;;
        --multi-nun-counts)          MULTI_NUN_COUNTS="$2";           shift 2 ;;
        --downsample)                DOWNSAMPLE="$2";                 shift 2 ;;
        --seed)                      SEED="$2";                       shift 2 ;;
        --exclude-glacier-long)          INCLUDE_GLACIER_LONG=0;          shift ;;
        --exclude-mascots-long)          INCLUDE_MASCOTS_LONG=0;          shift ;;
        --include-mascots-long)          INCLUDE_MASCOTS_LONG=1;          shift ;;
        --skip)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SKIP+=("$1"); shift
            done
            ;;
        --only)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ONLY+=("$1"); shift
            done
            ;;
        --help|-h)
            sed -n '/^# Usage/,/^[^#]/{ /^[^#]/d; s/^# \{0,1\}//; p }' "$0"
            exit 0
            ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve which experiments to run
# ---------------------------------------------------------------------------
declare -A TO_RUN
for name in "${ALL_EXPERIMENTS[@]}"; do
    TO_RUN[$name]=1
done

if [[ ${#ONLY[@]} -gt 0 ]]; then
    for name in "${ALL_EXPERIMENTS[@]}"; do
        TO_RUN[$name]=0
    done
    for name in "${ONLY[@]}"; do
        if [[ -z "${TO_RUN[$name]+x}" ]]; then
            echo "ERROR: unknown experiment in --only: $name" >&2; exit 1
        fi
        TO_RUN[$name]=1
    done
fi

for name in "${SKIP[@]}"; do
    if [[ -z "${TO_RUN[$name]+x}" ]]; then
        echo "ERROR: unknown experiment in --skip: $name" >&2; exit 1
    fi
    TO_RUN[$name]=0
done

# Default FruitFlies sample count matches --n-samples unless overridden.
if [[ -z "$FRUITFLIES_N_SAMPLES" ]]; then
    FRUITFLIES_N_SAMPLES="$N_SAMPLES"
fi

# Build exclude-flags array for long-series compare scripts (FaultDetectionA, FruitFlies).
# Pass --exclude-glacier-long / --exclude-mascots-long to disable them if OOM occurs.
LONG_SERIES_EXCLUDE_FLAGS=()
[[ "$INCLUDE_GLACIER_LONG" -eq 0 ]] && LONG_SERIES_EXCLUDE_FLAGS+=("--exclude-glacier")
[[ "$INCLUDE_MASCOTS_LONG" -eq 0 ]] && LONG_SERIES_EXCLUDE_FLAGS+=("--exclude-mascots")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

RESULTS=()   # "name:ok" entries
WALL_START=$SECONDS

run_experiment() {
    local name="$1"; shift
    local cmd=("$@")

    echo ""
    echo "======================================================================"
    echo "  [$name]"
    echo "  ${cmd[*]}"
    echo "======================================================================"
    echo ""

    local t0=$SECONDS
    if "${cmd[@]}"; then
        local status="OK"
        local ok=1
    else
        local status="FAILED (exit $?)"
        local ok=0
    fi
    local elapsed=$(( SECONDS - t0 ))
    echo ""
    echo "[$name] $status — ${elapsed}s"

    RESULTS+=("$name:$ok")
}

# ---------------------------------------------------------------------------
# Pre-create output directories
# ---------------------------------------------------------------------------
mkdir -p \
    "$OUT_DIR/faultdetectiona_ablation" \
    "$OUT_DIR/faultdetectiona_compare" \
    "$OUT_DIR/fruitflies_ablation" \
    "$OUT_DIR/fruitflies_compare"

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
if [[ "${TO_RUN[faultdetectiona_ablation]}" -eq 1 ]]; then
    run_experiment faultdetectiona_ablation \
        "$PYTHON" "$SCRIPT_DIR/faultdetectiona/ablation_faultdetectiona.py" \
        --max-samples "$ABLATION_MAX_SAMPLES" \
        --max-plot-samples "$ABLATION_MAX_PLOT_SAMPLES" \
        --multi-nun-counts "$MULTI_NUN_COUNTS" \
        --downsample "$DOWNSAMPLE" \
        --output-prefix "$OUT_DIR/faultdetectiona_ablation/faultdetectiona"
else
    echo ""
    echo "[faultdetectiona_ablation] SKIPPED"
fi

if [[ "${TO_RUN[faultdetectiona_compare]}" -eq 1 ]]; then
    run_experiment faultdetectiona_compare \
        "$PYTHON" "$SCRIPT_DIR/faultdetectiona/compare_faultdetectiona.py" \
        --n-samples "$N_SAMPLES" \
        --out-dir "$OUT_DIR/faultdetectiona_compare" \
        --seed "$SEED" \
        "${LONG_SERIES_EXCLUDE_FLAGS[@]}"
else
    echo ""
    echo "[faultdetectiona_compare] SKIPPED"
fi

if [[ "${TO_RUN[fruitflies_ablation]}" -eq 1 ]]; then
    run_experiment fruitflies_ablation \
        "$PYTHON" "$SCRIPT_DIR/fruitflies/ablation_fruitflies.py" \
        --max-samples "$ABLATION_MAX_SAMPLES" \
        --max-plot-samples "$ABLATION_MAX_PLOT_SAMPLES" \
        --multi-nun-counts "$MULTI_NUN_COUNTS" \
        --downsample "$DOWNSAMPLE" \
        --output-prefix "$OUT_DIR/fruitflies_ablation/fruitflies"
else
    echo ""
    echo "[fruitflies_ablation] SKIPPED"
fi

if [[ "${TO_RUN[fruitflies_compare]}" -eq 1 ]]; then
    run_experiment fruitflies_compare \
        "$PYTHON" "$SCRIPT_DIR/fruitflies/compare_fruitflies.py" \
        --n-samples "$FRUITFLIES_N_SAMPLES" \
        --out-dir "$OUT_DIR/fruitflies_compare" \
        --seed "$SEED" \
        "${LONG_SERIES_EXCLUDE_FLAGS[@]}"
else
    echo ""
    echo "[fruitflies_compare] SKIPPED"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(( SECONDS - WALL_START ))
echo ""
echo "======================================================================"
echo "  EXPERIMENT SUMMARY  (total ${TOTAL}s)"
echo "======================================================================"

FAILED=()
for entry in "${RESULTS[@]}"; do
    name="${entry%%:*}"
    ok="${entry##*:}"
    if [[ "$ok" -eq 1 ]]; then
        echo "  OK     $name"
    else
        echo "  FAILED $name"
        FAILED+=("$name")
    fi
done

for name in "${ALL_EXPERIMENTS[@]}"; do
    in_results=0
    for entry in "${RESULTS[@]}"; do
        [[ "${entry%%:*}" == "$name" ]] && in_results=1 && break
    done
    [[ "$in_results" -eq 0 ]] && echo "  SKIP   $name"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "${#FAILED[@]} experiment(s) failed: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "All outputs written under: $OUT_DIR"
