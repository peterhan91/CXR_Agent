#!/bin/bash
# Score predictions with all 7 ReXrank metrics.
# Usage: bash scripts/score_rexrank.sh results/eval_val/
#
# Expects: gt_*.csv and pred_*.csv files (from eval_mimic.py --mode score)
# Outputs: scores_*_rexrank.json with all 7 metrics

set -e

OUTPUT_DIR="$(cd "${1:?Usage: $0 <output_dir>}" && pwd)"
REXRANK_DIR="/home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric"
GREEN_DIR="/home/than/DeepLearning/GREEN"

echo "=== ReXrank Scoring Pipeline ==="
echo "Output dir: $OUTPUT_DIR"

# Find all GT/pred CSV pairs
for gt_csv in "$OUTPUT_DIR"/gt_*.csv; do
    [ -f "$gt_csv" ] || continue
    mode_name=$(basename "$gt_csv" | sed 's/^gt_//' | sed 's/\.csv$//')
    pred_csv="$OUTPUT_DIR/pred_${mode_name}.csv"
    [ -f "$pred_csv" ] || { echo "SKIP: no pred_${mode_name}.csv"; continue; }

    echo ""
    echo "--- Scoring: $mode_name ---"
    scores_file="$OUTPUT_DIR/scores_${mode_name}_rexrank.json"

    # Step 1: CXR-Report-Metric (5 core metrics) — radgraph env
    # Note: Must use a temp script file (not python -c) because calc_metric
    # uses pickle/multiprocessing which needs __main__ to be a real module.
    echo "[1/3] CXR-Report-Metric (BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ-v1)..."
    cxr_scores_file="$OUTPUT_DIR/scores_${mode_name}_cxr.json"
    tmp_script=$(mktemp /tmp/cxr_metric_XXXXXX.py)
    cat > "$tmp_script" <<PYEOF
# Import CompositeMetric into __main__ so pickle can find it
# (the model .pkl was saved from __main__ scope)
from CXRMetric.run_eval import CompositeMetric, calc_metric
calc_metric('$gt_csv', '$pred_csv', '$cxr_scores_file', use_idf=True)
print('CXR-Report-Metric done')
PYEOF
    PYTHONPATH="$REXRANK_DIR" conda run --cwd "$REXRANK_DIR" -n radgraph python "$tmp_script" 2>&1 | grep -v "^$"
    rm -f "$tmp_script"

    # Step 2: RaTEScore — green_score env
    echo "[2/3] RaTEScore..."
    conda run -n green_score python -c "
import json, csv
from RaTEScore import RaTEScore

scorer = RaTEScore()

gt_reports, pred_reports = [], []
with open('$gt_csv') as f:
    reader = csv.DictReader(f)
    gt_reports = [row['report'] for row in reader]
with open('$pred_csv') as f:
    reader = csv.DictReader(f)
    pred_reports = [row['report'] for row in reader]

scores = scorer.compute_score(gt_reports, pred_reports)
avg_score = sum(scores) / len(scores)
print(f'RaTEScore: {avg_score:.4f}')

with open('$OUTPUT_DIR/scores_${mode_name}_ratescore.json', 'w') as f:
    json.dump({'ratescore': round(avg_score, 4), 'per_study': [round(s, 4) for s in scores]}, f, indent=2)
" 2>&1 | grep -v "^$"

    # Step 3: GREEN — green_score env
    # CUDA_VISIBLE_DEVICES=1: use GPU 1 (most free VRAM); single GPU avoids dist.init_process_group
    echo "[3/3] GREEN score..."
    CUDA_VISIBLE_DEVICES=1 conda run --cwd "$GREEN_DIR" -n green_score python -c "
import json, csv
from green_score.green import GREEN

scorer = GREEN(model_name='StanfordAIMI/GREEN-radllama2-7b')
scorer.batch_size = 2  # reduce from 8 to avoid OOM

gt_reports, pred_reports = [], []
with open('$gt_csv') as f:
    reader = csv.DictReader(f)
    gt_reports = [row['report'] for row in reader]
with open('$pred_csv') as f:
    reader = csv.DictReader(f)
    pred_reports = [row['report'] for row in reader]

mean, std, green_scores, summary, result_df = scorer(refs=gt_reports, hyps=pred_reports)
print(f'GREEN: {mean:.4f} ± {std:.4f}')

with open('$OUTPUT_DIR/scores_${mode_name}_green.json', 'w') as f:
    json.dump({'green_score': round(mean, 4), 'green_std': round(std, 4), 'per_study': [round(s, 4) for s in green_scores]}, f, indent=2)
" 2>&1 | grep -v "^$"

    # Merge all scores into one file
    echo "[*] Merging scores..."
    python3 -c "
import json, csv
merged = {}

# CXR-Report-Metric outputs CSV, not JSON — parse and compute means
try:
    with open('$cxr_scores_file') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if rows:
            for col in ['bleu_score', 'bertscore', 'semb_score', 'radgraph_combined', 'RadCliQ-v0', 'RadCliQ-v1']:
                vals = [float(r[col]) for r in rows if col in r]
                if vals:
                    # Rename to match ReXrank convention
                    key = {'bleu_score': 'bleu2', 'bertscore': 'bertscore', 'semb_score': 'semb_score',
                           'radgraph_combined': 'radgraph_combined', 'RadCliQ-v0': 'radcliq_v0', 'RadCliQ-v1': 'radcliq_v1'}[col]
                    merged[key] = round(sum(vals) / len(vals), 4)
            print(f'CXR-Report-Metric: {len(rows)} studies')
except Exception as e:
    print(f'Warning: CXR scores: {e}')

# RaTEScore and GREEN output JSON
for f in ['$OUTPUT_DIR/scores_${mode_name}_ratescore.json', '$OUTPUT_DIR/scores_${mode_name}_green.json']:
    try:
        with open(f) as fh:
            data = json.load(fh)
            for k, v in data.items():
                if k != 'per_study':
                    merged[k] = v
    except Exception as e:
        print(f'Warning: {f}: {e}')

with open('$scores_file', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Merged scores -> $scores_file')
"

    echo "--- $mode_name complete ---"
    cat "$scores_file"
done

echo ""
echo "=== All scoring complete ==="
