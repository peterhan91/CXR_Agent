#!/bin/bash
# Run all remaining feature evaluations sequentially.
# Feature 5 (metadata) already done. Remaining: 1a, 1b, 4, 2, 3.
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

    if [ -f "$outdir/predictions_agent_initial.json" ]; then
        n_done=$($PYTHON -c "import json; print(len(json.load(open('$outdir/predictions_agent_initial.json'))))")
        n_total=$($PYTHON -c "import json; print(len(json.load(open('$EVAL_INPUT'))))")
        if [ "$n_done" = "$n_total" ]; then
            echo "Agent eval already complete ($n_done/$n_total). Skipping agent run."
        else
            echo "Resuming agent eval ($n_done/$n_total done)..."
            $PYTHON -u scripts/eval_mimic.py --mode agent \
                --input "$EVAL_INPUT" \
                --output "$outdir" \
                --config "$config"
        fi
    else
        echo "Starting agent eval..."
        $PYTHON -u scripts/eval_mimic.py --mode agent \
            --input "$EVAL_INPUT" \
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

echo "Starting feature evaluation runs at $(date)"
echo "============================================="

# Feature 1a: MedGemma grounding (additive)
run_feature "feat1a_grounding_add" "configs/config_feat1a_grounding_add.yaml"

# Feature 1b: MedGemma grounding (replacement)
run_feature "feat1b_grounding_replace" "configs/config_feat1b_grounding_replace.yaml"

# Feature 4: Multi-view input
run_feature "feat4_multiview" "configs/config_feat4_multiview.yaml"

# Feature 2: MedGemma longitudinal
run_feature "feat2_longitudinal_mg" "configs/config_feat2_longitudinal_mg.yaml"

# Feature 3: CheXagent-2 temporal
run_feature "feat3_temporal_ca2" "configs/config_feat3_temporal_ca2.yaml"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL FEATURE EVALS COMPLETE                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Finished at $(date)"

# Print comparison table
$PYTHON -c "
import json, os

scores_root = '$EVAL_DIR'
features = ['baseline', 'feat5_metadata', 'feat1a_grounding_add', 'feat1b_grounding_replace',
            'feat4_multiview', 'feat2_longitudinal_mg', 'feat3_temporal_ca2']

print()
print('Feature Comparison (findings):')
print(f'{\"Feature\":<30} {\"RadGraph F1\":>12} {\"1/RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 70)
for feat in features:
    fpath = os.path.join(scores_root, feat, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{feat:<30} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('findings', {})
    if not s:
        print(f'{feat:<30} (no findings scores)')
        continue
    print(f'{feat:<30} {s[\"RadGraph_F1\"]:>12.4f} {s[\"1/RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')

print()
print('Feature Comparison (reports):')
print(f'{\"Feature\":<30} {\"RadGraph F1\":>12} {\"1/RadCliQ\":>10} {\"BLEU\":>8} {\"BERT\":>8}')
print('-' * 70)
for feat in features:
    fpath = os.path.join(scores_root, feat, 'scores', 'summary_fast.json')
    if not os.path.exists(fpath):
        print(f'{feat:<30} (not scored)')
        continue
    with open(fpath) as f:
        data = json.load(f)
    s = data.get('agent_initial', {}).get('reports', {})
    if not s:
        print(f'{feat:<30} (no reports scores)')
        continue
    print(f'{feat:<30} {s[\"RadGraph_F1\"]:>12.4f} {s[\"1/RadCliQ_v1\"]:>10.4f} {s[\"BLEU\"]:>8.4f} {s[\"BERT\"]:>8.4f}')
print()
"
