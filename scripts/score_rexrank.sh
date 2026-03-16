#!/bin/bash
# Score predictions with all 7 ReXrank metrics.
#
# Expects the directory structure created by: eval_mimic.py --mode score
#   {output_dir}/scores/{model}/{section}/gt*.csv + pred*.csv
#
# Usage:
#   bash scripts/score_rexrank.sh results/eval_baseline
#
# Scores every gt/pred CSV pair found under scores/{model}/{section}/ and
# writes a merged JSON next to each pair.  After all pairs are scored,
# produces a summary table at {output_dir}/scores/summary.json + .txt
#
# Environments required:
#   radgraph    — CXR-Report-Metric (BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ)
#   green_score — RaTEScore + GREEN

set -e

OUTPUT_DIR="$(cd "${1:?Usage: $0 <output_dir>}" && pwd)"
SCORES_DIR="$OUTPUT_DIR/scores"
REXRANK_DIR="/home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric"
GREEN_DIR="/home/than/DeepLearning/GREEN"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: $SCORES_DIR not found. Run eval_mimic.py --mode score first."
    exit 1
fi

echo "=== ReXrank Scoring Pipeline ==="
echo "Scores dir: $SCORES_DIR"
echo ""

# ── Helper: score one gt/pred CSV pair ──────────────────────────────────────
score_pair() {
    local gt_csv="$1"
    local pred_csv="$2"
    local out_dir="$(dirname "$gt_csv")"
    local tag="$3"   # e.g. "overall" or "mimic_cxr"

    local cxr_csv="$out_dir/cxr_report_scores_${tag}.csv"
    local rate_json="$out_dir/ratescore_${tag}.json"
    local green_json="$out_dir/green_${tag}.json"
    local merged_json="$out_dir/scores_${tag}.json"

    local n_studies
    n_studies=$(tail -n +2 "$gt_csv" | wc -l)
    echo "  [$tag] $n_studies studies"

    # Step 1: CXR-Report-Metric (5 core metrics) — radgraph env
    echo "    [1/3] CXR-Report-Metric..."
    local tmp_script
    tmp_script=$(mktemp /tmp/cxr_metric_XXXXXX.py)
    cat > "$tmp_script" <<PYEOF
from CXRMetric.run_eval import CompositeMetric, calc_metric
calc_metric('$gt_csv', '$pred_csv', '$cxr_csv', use_idf=True)
PYEOF
    PYTHONPATH="$REXRANK_DIR" conda run --cwd "$REXRANK_DIR" -n radgraph python "$tmp_script" 2>&1 | tail -1
    rm -f "$tmp_script"

    # Step 2: RaTEScore — green_score env
    echo "    [2/3] RaTEScore..."
    conda run -n green_score python -c "
import json, csv
from RaTEScore import RaTEScore
scorer = RaTEScore()
with open('$gt_csv') as f:
    gt = [row['report'] for row in csv.DictReader(f)]
with open('$pred_csv') as f:
    pred = [row['report'] for row in csv.DictReader(f)]
scores = scorer.compute_score(gt, pred)
avg = sum(scores) / len(scores)
print(f'  RaTEScore: {avg:.4f}')
with open('$rate_json', 'w') as f:
    json.dump({'ratescore': round(avg, 4), 'per_study': [round(s, 4) for s in scores]}, f, indent=2)
" 2>&1 | grep -v "^$"

    # Step 3: GREEN — green_score env
    echo "    [3/3] GREEN..."
    CUDA_VISIBLE_DEVICES=1 conda run --cwd "$GREEN_DIR" -n green_score python -c "
import json, csv, re, numpy as np

def sanitize(text):
    \"\"\"Strip non-ASCII and control chars that cause tokenizer index-out-of-range.\"\"\"
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    return text.strip() or 'No report available.'

with open('$gt_csv') as f:
    gt = [sanitize(row['report']) for row in csv.DictReader(f)]
with open('$pred_csv') as f:
    pred = [sanitize(row['report']) for row in csv.DictReader(f)]

from green_score.green import GREEN
scorer = GREEN(model_name='StanfordAIMI/GREEN-radllama2-7b')
scorer.batch_size = 1  # batch_size=1 avoids cross-contamination on CUDA errors

import torch
green_scores = []
for i, (g, p) in enumerate(zip(gt, pred)):
    try:
        _, _, scores, _, _ = scorer(refs=[g], hyps=[p])
        green_scores.append(scores[0])
    except (RuntimeError, torch.cuda.CudaError) as e:
        print(f'  GREEN: study {i} failed ({e.__class__.__name__}), assigning 0.0')
        green_scores.append(0.0)
        torch.cuda.empty_cache()
        # Reload model after CUDA error corrupts state
        scorer = GREEN(model_name='StanfordAIMI/GREEN-radllama2-7b')
        scorer.batch_size = 1

arr = np.array(green_scores)
mean, std = float(arr.mean()), float(arr.std())
print(f'  GREEN: {mean:.4f} +/- {std:.4f}')
with open('$green_json', 'w') as f:
    json.dump({'green_score': round(mean, 4), 'green_std': round(std, 4), 'per_study': [round(s, 4) for s in green_scores]}, f, indent=2)
" 2>&1 | grep -v "^$"

    # Merge all scores into one JSON
    python3 -c "
import json, csv
merged = {'n_studies': $n_studies}

# CXR-Report-Metric (CSV output)
try:
    with open('$cxr_csv') as fh:
        rows = list(csv.DictReader(fh))
    for col, key in [('bleu_score','BLEU'), ('bertscore','BERT'), ('semb_score','Semb'),
                     ('radgraph_combined','RadG'), ('RadCliQ-v1','RCliQ')]:
        vals = [float(r[col]) for r in rows if col in r]
        if vals: merged[key] = round(sum(vals)/len(vals), 4)
except Exception as e:
    print(f'  Warning CXR: {e}')

# RaTEScore
try:
    with open('$rate_json') as fh:
        merged['RaTE'] = json.load(fh)['ratescore']
except Exception as e:
    print(f'  Warning RaTE: {e}')

# GREEN
try:
    with open('$green_json') as fh:
        merged['GREEN'] = json.load(fh)['green_score']
except Exception as e:
    print(f'  Warning GREEN: {e}')

with open('$merged_json', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'    -> {merged}')
"
}

# ── Main loop: iterate scores/{model}/{section}/ ───────────────────────────

for model_dir in "$SCORES_DIR"/*/; do
    model=$(basename "$model_dir")
    echo "━━━ Model: $model ━━━"

    for section_dir in "$model_dir"*/; do
        section=$(basename "$section_dir")
        echo ""
        echo "── $model / $section ──"

        # Overall (all datasets combined)
        gt_all="$section_dir/gt.csv"
        pred_all="$section_dir/pred.csv"
        if [ -f "$gt_all" ] && [ -f "$pred_all" ]; then
            score_pair "$gt_all" "$pred_all" "overall"
        fi

        # Per-dataset
        for gt_ds in "$section_dir"/gt_*.csv; do
            [ -f "$gt_ds" ] || continue
            ds_name=$(basename "$gt_ds" | sed 's/^gt_//' | sed 's/\.csv$//')
            pred_ds="$section_dir/pred_${ds_name}.csv"
            [ -f "$pred_ds" ] || continue
            score_pair "$gt_ds" "$pred_ds" "$ds_name"
        done
    done
    echo ""
done

# ── Aggregate summary ──────────────────────────────────────────────────────
echo "━━━ Generating summary ━━━"
SCORES_DIR="$SCORES_DIR" python3 << 'PYEOF'
import json, os

scores_dir = os.environ["SCORES_DIR"]

rows = []
for model in sorted(os.listdir(scores_dir)):
    model_dir = os.path.join(scores_dir, model)
    if not os.path.isdir(model_dir):
        continue
    for section in sorted(os.listdir(model_dir)):
        section_dir = os.path.join(model_dir, section)
        if not os.path.isdir(section_dir):
            continue
        for fn in sorted(os.listdir(section_dir)):
            if not fn.startswith("scores_") or not fn.endswith(".json"):
                continue
            dataset = fn.replace("scores_", "").replace(".json", "")
            with open(os.path.join(section_dir, fn)) as f:
                scores = json.load(f)
            rows.append({
                "model": model,
                "section": section,
                "dataset": dataset,
                **{k: v for k, v in scores.items() if k != "per_study"},
            })

# Save JSON
summary_json = os.path.join(scores_dir, "summary.json")
with open(summary_json, "w") as f:
    json.dump(rows, f, indent=2)

# Print + save text table
metrics = ["BLEU", "BERT", "Semb", "RadG", "RCliQ", "RaTE", "GREEN"]
header = f"{'Model':<12} {'Section':<10} {'Dataset':<16} {'N':>4}  " + "  ".join(f"{m:>6}" for m in metrics)
sep = "-" * len(header)

lines = [header, sep]
for r in rows:
    vals = "  ".join(f"{r.get(m, 0):>6.4f}" if isinstance(r.get(m), (int, float)) else f"{'---':>6}" for m in metrics)
    lines.append(f"{r['model']:<12} {r['section']:<10} {r['dataset']:<16} {r.get('n_studies','?'):>4}  {vals}")
lines.append(sep)

table = "\n".join(lines)
print(table)

summary_txt = os.path.join(scores_dir, "summary.txt")
with open(summary_txt, "w") as f:
    f.write(table + "\n")

print(f"\nSaved: {summary_json}")
print(f"Saved: {summary_txt}")
PYEOF

echo ""
echo "=== All scoring complete ==="
