#!/bin/bash
# Run combo feature evaluations:
#   1. combo_baseline (f5_meta + f1b_grounding) on BOTH test sets
#   2. combo_multiview (baseline + lateral) on multiview test set
#   3. combo_temporal (baseline + temporal) on temporal test set
#
# Usage: nohup bash scripts/run_combo_evals.sh > logs/combo_evals.log 2>&1 &
set -e

EVAL_DIR="results/eval_combo"
PYTHON="/home/than/anaconda3/envs/cxr_agent/bin/python"
MV_INPUT="data/eval/combo_test/multiview_20.json"
TP_INPUT="data/eval/combo_test/temporal_20.json"

run_eval() {
    local name="$1"
    local config="$2"
    local input="$3"
    local outdir="$EVAL_DIR/$name"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Eval: $name"
    echo "║  Config: $config"
    echo "║  Input:  $input"
    echo "║  Output: $outdir"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    if [ -f "$outdir/predictions_agent_initial.json" ]; then
        n_done=$($PYTHON -c "import json; print(len(json.load(open('$outdir/predictions_agent_initial.json'))))")
        n_total=$($PYTHON -c "import json; print(len(json.load(open('$input'))))")
        if [ "$n_done" = "$n_total" ]; then
            echo "Agent eval already complete ($n_done/$n_total). Skipping agent run."
        else
            echo "Resuming agent eval ($n_done/$n_total done)..."
            $PYTHON -u scripts/eval_mimic.py --mode agent \
                --input "$input" \
                --output "$outdir" \
                --config "$config"
        fi
    else
        echo "Starting agent eval..."
        $PYTHON -u scripts/eval_mimic.py --mode agent \
            --input "$input" \
            --output "$outdir" \
            --config "$config"
    fi

    echo "Exporting score CSVs..."
    $PYTHON -u scripts/eval_mimic.py --mode score --output "$outdir"

    echo "Running fast scoring (CXR-Report-Metric)..."
    bash scripts/score_fast.sh "$outdir"

    echo ""
    echo "=== $name DONE ==="
    echo ""
}

echo "Starting combo feature evaluation at $(date)"
echo "============================================="

# --- Multiview test set ---
# Baseline on multiview studies (no lateral passed — config has no multiview)
run_eval "mv_baseline" "configs/config_combo_baseline.yaml" "$MV_INPUT"

# Multiview on same studies (lateral passed)
run_eval "mv_multiview" "configs/config_combo_multiview.yaml" "$MV_INPUT"

# --- Temporal test set ---
# Baseline on temporal studies (no temporal tool)
run_eval "tp_baseline" "configs/config_combo_baseline.yaml" "$TP_INPUT"

# Temporal on same studies (temporal tool enabled)
run_eval "tp_temporal" "configs/config_combo_temporal.yaml" "$TP_INPUT"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL COMBO EVALS COMPLETE                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Finished at $(date)"

# Print comparison
$PYTHON -c "
import json, os

scores_root = '$EVAL_DIR'
evals = ['mv_baseline', 'mv_multiview', 'tp_baseline', 'tp_temporal']

print()
print('Combo Feature Comparison (findings):')
print(f'{\"Eval\":<20} {\"RadGraph F1\":>12} {\"1/RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 62)
for name in evals:
    fpath = os.path.join(scores_root, name, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{name:<20} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('findings', {})
    if not s:
        print(f'{name:<20} (no findings scores)')
        continue
    print(f'{name:<20} {s[\"RadGraph_F1\"]:>12.4f} {s[\"1/RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')

print()
print('Combo Feature Comparison (reports):')
print(f'{\"Eval\":<20} {\"RadGraph F1\":>12} {\"1/RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 62)
for name in evals:
    fpath = os.path.join(scores_root, name, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{name:<20} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('reports', {})
    if not s:
        print(f'{name:<20} (no reports scores)')
        continue
    print(f'{name:<20} {s[\"RadGraph_F1\"]:>12.4f} {s[\"1/RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')
print()
"
