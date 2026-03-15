#!/bin/bash
# Run GREEN score + Aggregate for all eval results.
# Steps 1 (CXR-Report-Metric) and 2 (RaTEScore) already completed.
#
# Usage:
#   nohup bash scripts/run_rexrank_green.sh > logs/rexrank_green.log 2>&1 &

set -e
eval "$(conda shell.bash hook)"

REXRANK_ROOT="/home/than/DeepLearning/ReXrank-metric"
CXR_AGENT_ROOT="/home/than/DeepLearning/CXR_Agent"
GREEN_ROOT="/home/than/DeepLearning/GREEN"
REXRANK_DATA="$REXRANK_ROOT/data"
REXRANK_RESULTS="$REXRANK_ROOT/results"
REXRANK_SCRIPTS="$REXRANK_ROOT/scripts"
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU

DATASETS="eval_baseline eval_followup eval_baseline_noclear eval_followup_noclear"

conda activate green_score

# ============================================================
# Step 3: GREEN Score (must cd into GREEN repo for imports)
# ============================================================

echo "=== Step 3: GREEN Score ==="
export PYTHONPATH="$GREEN_ROOT:$PYTHONPATH"

for dataset in $DATASETS; do
    MODELS=""
    for m in agent chexone medversa; do
        if [ -f "$REXRANK_DATA/reports/$dataset/gt_reports_${m}.csv" ]; then
            MODELS="$MODELS $m"
        fi
    done
    MODELS=$(echo $MODELS | xargs)

    if [ -z "$MODELS" ]; then continue; fi

    echo "[RUN] GREEN: $dataset [$MODELS]"
    cd "$GREEN_ROOT"
    python "$REXRANK_SCRIPTS/run_green.py" \
        --datasets "$dataset" \
        --models $MODELS \
        --splits reports \
        --data-root "$REXRANK_DATA" \
        --results-root "$REXRANK_RESULTS"
    cd "$CXR_AGENT_ROOT"
done

echo "=== Step 3 complete ==="
echo ""

# ============================================================
# Step 4: Aggregate metrics
# ============================================================

echo "=== Step 4: Aggregate ==="

for dataset in $DATASETS; do
    MODELS=""
    for m in agent chexone medversa; do
        if [ -f "$REXRANK_DATA/reports/$dataset/gt_reports_${m}.csv" ]; then
            MODELS="$MODELS $m"
        fi
    done
    MODELS=$(echo $MODELS | xargs)

    if [ -z "$MODELS" ]; then continue; fi

    echo "[RUN] Aggregate: $dataset [$MODELS]"
    python "$REXRANK_SCRIPTS/aggregate_metrics.py" \
        --datasets "$dataset" \
        --models $MODELS \
        --splits reports \
        --results-root "$REXRANK_RESULTS" \
        --output-root "$REXRANK_RESULTS/metric"
done

echo "=== Step 4 complete ==="
echo ""

# ============================================================
# Step 5: Copy aggregated results back to CXR_Agent
# ============================================================

echo "=== Step 5: Copy results back ==="

copy_results() {
    local dataset="$1"
    local eval_dir="$2"

    # Copy per-sample scores
    local scores_dir="$REXRANK_RESULTS/${dataset}_reports"
    if [ -d "$scores_dir" ]; then
        for f in "$scores_dir"/report_scores_*.csv "$scores_dir"/ratescore_*.csv "$scores_dir"/results_green_*.csv 2>/dev/null; do
            [ -f "$f" ] && cp "$f" "$eval_dir/" && echo "[COPY] $(basename $f) -> $eval_dir/"
        done
    fi

    # Copy aggregated metrics
    local metric_dir="$REXRANK_RESULTS/metric/${dataset}/reports"
    if [ -d "$metric_dir" ]; then
        for f in "$metric_dir"/*.csv 2>/dev/null; do
            [ -f "$f" ] && cp "$f" "$eval_dir/rexrank_$(basename $f)" && echo "[COPY] rexrank_$(basename $f) -> $eval_dir/"
        done
    fi
}

copy_results "eval_baseline" "$CXR_AGENT_ROOT/results/eval_baseline"
copy_results "eval_followup" "$CXR_AGENT_ROOT/results/eval_followup"
copy_results "eval_baseline_noclear" "$CXR_AGENT_ROOT/results/eval_baseline_noclear"
copy_results "eval_followup_noclear" "$CXR_AGENT_ROOT/results/eval_followup_noclear"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
