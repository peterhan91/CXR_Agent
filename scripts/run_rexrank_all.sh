#!/bin/bash
# ============================================================
# Run ReXrank metrics for all 16-bit fix eval results
# ============================================================
#
# Evaluates: agent, chexone, medversa across 4 tracks:
#   eval_baseline, eval_followup, eval_baseline_noclear, eval_followup_noclear
#
# Usage:
#   nohup bash scripts/run_rexrank_all.sh > logs/rexrank_all.log 2>&1 &
# ============================================================

set -e
eval "$(conda shell.bash hook)"

REXRANK_ROOT="/home/than/DeepLearning/ReXrank-metric"
CXR_AGENT_ROOT="/home/than/DeepLearning/CXR_Agent"
REXRANK_DATA="$REXRANK_ROOT/data"
REXRANK_RESULTS="$REXRANK_ROOT/results"
REXRANK_SCRIPTS="$REXRANK_ROOT/scripts"
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU

# ============================================================
# Step 0: Convert our CSVs (study_id,report) to ReXrank format
#         (study_id,case_id,report) and copy into ReXrank data/
# ============================================================

echo "=== Step 0: Prepare CSVs for ReXrank ==="

prepare_csv() {
    local src_gt="$1"
    local src_pred="$2"
    local dataset="$3"
    local model="$4"

    if [ ! -f "$src_gt" ] || [ ! -f "$src_pred" ]; then
        echo "[SKIP] Missing: $src_gt or $src_pred"
        return
    fi

    local out_dir="$REXRANK_DATA/reports/$dataset"
    mkdir -p "$out_dir"

    # Add case_id column (same as study_id) if not present
    python3 -c "
import pandas as pd, sys
gt = pd.read_csv('$src_gt')
pred = pd.read_csv('$src_pred')
if 'case_id' not in gt.columns:
    gt.insert(1, 'case_id', gt['study_id'])
if 'case_id' not in pred.columns:
    pred.insert(1, 'case_id', pred['study_id'])
# Re-index study_id as integers
gt['study_id'] = range(len(gt))
pred['study_id'] = range(len(pred))
gt.to_csv('$out_dir/gt_reports_${model}.csv', index=False)
pred.to_csv('$out_dir/predicted_reports_${model}.csv', index=False)
print(f'[OK] {len(gt)} rows -> $out_dir/*_${model}.csv')
"
}

# --- eval_baseline: agent, chexone, medversa ---
for model in agent chexone medversa; do
    prepare_csv \
        "$CXR_AGENT_ROOT/results/eval_baseline/gt_${model}.csv" \
        "$CXR_AGENT_ROOT/results/eval_baseline/pred_${model}.csv" \
        "eval_baseline" "$model"
done

# --- eval_followup: agent, chexone, medversa ---
for model in agent chexone medversa; do
    prepare_csv \
        "$CXR_AGENT_ROOT/results/eval_followup/gt_${model}.csv" \
        "$CXR_AGENT_ROOT/results/eval_followup/pred_${model}.csv" \
        "eval_followup" "$model"
done

# --- eval_baseline_noclear: agent only ---
prepare_csv \
    "$CXR_AGENT_ROOT/results/eval_baseline_noclear/gt_agent.csv" \
    "$CXR_AGENT_ROOT/results/eval_baseline_noclear/pred_agent.csv" \
    "eval_baseline_noclear" "agent"

# --- eval_followup_noclear: agent only ---
prepare_csv \
    "$CXR_AGENT_ROOT/results/eval_followup_noclear/gt_agent.csv" \
    "$CXR_AGENT_ROOT/results/eval_followup_noclear/pred_agent.csv" \
    "eval_followup_noclear" "agent"

echo ""
echo "=== Step 0 complete ==="
echo ""

# ============================================================
# Step 1: CXR-Report-Metric (radgraph env)
#   BLEU, BERTScore, SembScore, RadGraph, RadCliQ
# ============================================================

echo "=== Step 1: CXR-Report-Metric (conda: radgraph) ==="
conda activate radgraph

DATASETS="eval_baseline eval_followup eval_baseline_noclear eval_followup_noclear"

for dataset in $DATASETS; do
    # Determine which models exist for this dataset
    MODELS=""
    for m in agent chexone medversa; do
        if [ -f "$REXRANK_DATA/reports/$dataset/gt_reports_${m}.csv" ]; then
            MODELS="$MODELS $m"
        fi
    done
    MODELS=$(echo $MODELS | xargs)  # trim

    if [ -z "$MODELS" ]; then
        echo "[SKIP] No models found for $dataset"
        continue
    fi

    echo "[RUN] $dataset: models=[$MODELS]"
    python "$REXRANK_SCRIPTS/run_cxr_metrics.py" \
        --datasets "$dataset" \
        --models $MODELS \
        --splits reports \
        --data-root "$REXRANK_DATA" \
        --results-root "$REXRANK_RESULTS"
done

echo "=== Step 1 complete ==="
echo ""

# ============================================================
# Step 2: RaTEScore (green_score env)
# ============================================================

echo "=== Step 2: RaTEScore (conda: green_score) ==="
conda activate green_score

for dataset in $DATASETS; do
    MODELS=""
    for m in agent chexone medversa; do
        if [ -f "$REXRANK_DATA/reports/$dataset/gt_reports_${m}.csv" ]; then
            MODELS="$MODELS $m"
        fi
    done
    MODELS=$(echo $MODELS | xargs)

    if [ -z "$MODELS" ]; then continue; fi

    echo "[RUN] RaTEScore: $dataset [$MODELS]"
    python "$REXRANK_SCRIPTS/run_ratescore.py" \
        --datasets "$dataset" \
        --models $MODELS \
        --splits reports \
        --data-root "$REXRANK_DATA" \
        --results-root "$REXRANK_RESULTS"
done

echo "=== Step 2 complete ==="
echo ""

# ============================================================
# Step 3: GREEN Score (green_score env)
# ============================================================

echo "=== Step 3: GREEN Score ==="

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
    python "$REXRANK_SCRIPTS/run_green.py" \
        --datasets "$dataset" \
        --models $MODELS \
        --splits reports \
        --data-root "$REXRANK_DATA" \
        --results-root "$REXRANK_RESULTS"
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

    local metric_dir="$REXRANK_RESULTS/metric/${dataset}_reports"
    if [ ! -d "$metric_dir" ]; then
        metric_dir="$REXRANK_RESULTS/metric/${dataset}"
        if [ ! -d "$metric_dir" ]; then
            echo "[SKIP] No metric dir for $dataset"
            return
        fi
    fi

    for f in "$metric_dir"/*.csv "$metric_dir"/*.json 2>/dev/null; do
        [ -f "$f" ] && cp "$f" "$eval_dir/" && echo "[COPY] $(basename $f) -> $eval_dir/"
    done

    # Also copy per-sample scores
    local scores_dir="$REXRANK_RESULTS/${dataset}_reports"
    if [ -d "$scores_dir" ]; then
        for f in "$scores_dir"/report_scores_*.csv "$scores_dir"/ratescore_*.csv "$scores_dir"/results_green_*.csv 2>/dev/null; do
            [ -f "$f" ] && cp "$f" "$eval_dir/" && echo "[COPY] $(basename $f) -> $eval_dir/"
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
