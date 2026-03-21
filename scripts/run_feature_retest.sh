#!/bin/bash
# Rerun feature evals with fixed context gating: baseline, feat5, feat3, feat4
# Usage: nohup bash scripts/run_feature_retest.sh > logs/feature_retest.log 2>&1 &
set -e

EVAL_INPUT="data/eval/feature_test/all_20.json"
EVAL_DIR="results/eval_features"
PYTHON="/home/than/anaconda3/envs/cxr_agent/bin/python"

run_feature() {
    local name="$1"
    local config="$2"
    local outdir="$EVAL_DIR/$name"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Feature: $name"
    echo "║  Config:  $config"
    echo "║  Output:  $outdir"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    # Agent eval
    $PYTHON -u scripts/eval_mimic.py --mode agent \
        --input "$EVAL_INPUT" \
        --output "$outdir" \
        --config "$config"

    # Score
    echo "Exporting score CSVs..."
    $PYTHON -u scripts/eval_mimic.py --mode score --output "$outdir"

    echo "Running fast scoring (CXR-Report-Metric)..."
    bash scripts/score_fast.sh "$outdir"

    echo ""
    echo "=== $name DONE ==="
    echo ""
}

echo "Starting feature retest at $(date)"
echo "============================================="

run_feature "baseline"          "configs/config_initial.yaml"
run_feature "feat5_metadata"    "configs/config_feat5_metadata.yaml"
run_feature "feat3_temporal_ca2" "configs/config_feat3_temporal_ca2.yaml"
run_feature "feat4_multiview"   "configs/config_feat4_multiview.yaml"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL FEATURE RETESTS COMPLETE                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Finished at $(date)"

# Print comparison table
$PYTHON -c "
import json, os

scores_root = '$EVAL_DIR'
features = ['baseline', 'feat5_metadata', 'feat3_temporal_ca2', 'feat4_multiview',
            'feat1a_grounding_add', 'feat1b_grounding_replace']

print()
print('Feature Comparison (findings):')
print(f'{\"Feature\":<25} {\"RadGraph F1\":>12} {\"RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 65)
for feat in features:
    fpath = os.path.join(scores_root, feat, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{feat:<25} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('findings', {})
    if not s:
        print(f'{feat:<25} (no findings scores)')
        continue
    print(f'{feat:<25} {s[\"RadGraph_F1\"]:>12.4f} {s[\"RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')

print()
print('Feature Comparison (reports):')
print(f'{\"Feature\":<25} {\"RadGraph F1\":>12} {\"RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 65)
for feat in features:
    fpath = os.path.join(scores_root, feat, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{feat:<25} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('reports', {})
    if not s:
        print(f'{feat:<25} (no reports scores)')
        continue
    print(f'{feat:<25} {s[\"RadGraph_F1\"]:>12.4f} {s[\"RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')
print()
"

# --- Combo Full: all clinical features ---
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Combo Full: all clinical features           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
run_feature "combo_full" "configs/config_combo_full.yaml"
