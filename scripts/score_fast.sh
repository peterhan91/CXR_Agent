#!/bin/bash
# Fast scoring: CXR-Report-Metric only (RadGraph F1 + RadCliQ-v1).
# Skips RaTEScore and GREEN for quick feature gating.
#
# Usage:
#   bash scripts/score_fast.sh results/eval_features/baseline

set -e

OUTPUT_DIR="$(cd "${1:?Usage: $0 <output_dir>}" && pwd)"
SCORES_DIR="$OUTPUT_DIR/scores"
REXRANK_DIR="/home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: $SCORES_DIR not found. Run eval_mimic.py --mode score first."
    exit 1
fi

echo "=== Fast Scoring (CXR-Report-Metric only) ==="
echo "Scores dir: $SCORES_DIR"
echo ""

for model_dir in "$SCORES_DIR"/*/; do
    model=$(basename "$model_dir")
    echo "━━━ Model: $model ━━━"

    for section_dir in "$model_dir"/*/; do
        section=$(basename "$section_dir")
        echo ""
        echo "── $model / $section ──"

        # Score each gt/pred pair
        for gt_csv in "$section_dir"/gt*.csv; do
            [ -f "$gt_csv" ] || continue
            tag=$(basename "$gt_csv" .csv | sed 's/^gt_\?//')
            [ -z "$tag" ] && tag="overall"
            pred_csv="$section_dir/pred_${tag}.csv"
            [ "$tag" = "overall" ] && pred_csv="$section_dir/pred.csv"
            [ -f "$pred_csv" ] || continue

            cxr_csv="$section_dir/cxr_report_scores_${tag}.csv"

            n_studies=$(python3 -c "import csv; print(sum(1 for _ in csv.DictReader(open('$gt_csv'))))")
            echo "  [$tag] $n_studies studies"

            if [ -f "$cxr_csv" ]; then
                echo "    Already scored, skipping."
                continue
            fi

            tmp_script=$(mktemp /tmp/cxr_metric_XXXXXX.py)
            cat > "$tmp_script" <<PYEOF
from CXRMetric.run_eval import CompositeMetric, calc_metric
calc_metric('$gt_csv', '$pred_csv', '$cxr_csv', use_idf=True)
PYEOF
            PYTHONPATH="$REXRANK_DIR" conda run --cwd "$REXRANK_DIR" -n radgraph python "$tmp_script" 2>&1 | tail -1
            rm -f "$tmp_script"
        done
    done
    echo ""
done

# Summary
echo "=== Summary ==="
python3 -c "
import csv, os, statistics, json

scores_dir = '$SCORES_DIR'
summary = {}
for model in sorted(os.listdir(scores_dir)):
    model_dir = os.path.join(scores_dir, model)
    if not os.path.isdir(model_dir):
        continue
    summary[model] = {}
    for section in sorted(os.listdir(model_dir)):
        section_dir = os.path.join(model_dir, section)
        if not os.path.isdir(section_dir):
            continue
        overall_csv = os.path.join(section_dir, 'cxr_report_scores_overall.csv')
        if not os.path.exists(overall_csv):
            continue
        with open(overall_csv) as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        radg = statistics.mean([float(r['radgraph_combined']) for r in rows])
        rcliq = statistics.mean([float(r['RadCliQ-v1']) for r in rows])
        inv_rcliq = statistics.mean([1.0/float(r['RadCliQ-v1']) if float(r['RadCliQ-v1']) != 0 else 0 for r in rows])
        bleu = statistics.mean([float(r['bleu_score']) for r in rows])
        bert = statistics.mean([float(r['bertscore']) for r in rows])
        summary[model][section] = {
            'n': n, 'RadGraph_F1': round(radg, 4), 'RadCliQ_v1': round(rcliq, 4),
            '1/RadCliQ_v1': round(inv_rcliq, 4), 'BLEU': round(bleu, 4), 'BERT': round(bert, 4),
        }
        print(f'{model}/{section} (n={n}): RadGraph={radg:.4f}  RadCliQ={rcliq:.4f}  1/RadCliQ={inv_rcliq:.4f}  BLEU={bleu:.4f}')

with open(os.path.join(scores_dir, 'summary_fast.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary written to {scores_dir}/summary_fast.json')
"
