#!/bin/bash
# Full v5.1 eval pipeline — skill v5.1 (two-stage VQA confirmation)
# Run: nohup bash scripts/run_eval_v51.sh > logs/eval_v51.log 2>&1 &
set -e

OUT=results/eval_v51
INPUT=data/eval/sample_v51.json
CONFIG=configs/config_grounded.yaml
PY=/home/than/anaconda3/envs/cxr_agent/bin/python

echo "=========================================="
echo "CXR Agent v5.1 Evaluation Pipeline"
echo "Output: $OUT"
echo "Input:  $INPUT"
echo "Started: $(date)"
echo "=========================================="

# 1. Baselines (sequential — each ~10min)
echo ""
echo "=== Step 1/6: Baselines ==="

echo "[$(date +%H:%M:%S)] Running CheXOne baseline..."
$PY scripts/eval_mimic.py --mode chexone --output $OUT --input $INPUT

echo "[$(date +%H:%M:%S)] Running CheXagent-2 baseline..."
$PY scripts/eval_mimic.py --mode chexagent2 --output $OUT --input $INPUT

echo "[$(date +%H:%M:%S)] Running MedVersa baseline..."
$PY scripts/eval_mimic.py --mode medversa --output $OUT --input $INPUT

# 2. Agent: baseline track (60 studies, ~75min)
echo ""
echo "=== Step 2/6: Agent baseline track (60 studies) ==="
echo "[$(date +%H:%M:%S)] Starting agent baseline..."
$PY scripts/eval_mimic.py --mode agent --output $OUT --input $INPUT --track baseline --config $CONFIG

# 3. Agent: followup track (30 studies, ~40min)
echo ""
echo "=== Step 3/6: Agent followup track (30 studies) ==="
echo "[$(date +%H:%M:%S)] Starting agent followup..."
$PY scripts/eval_mimic.py --mode agent --output $OUT --input $INPUT --track followup --config $CONFIG

# 4. Prepare score CSVs
echo ""
echo "=== Step 4/6: Prepare score CSVs ==="
echo "[$(date +%H:%M:%S)] Scoring..."
$PY scripts/eval_mimic.py --mode score --output $OUT

# 5. ReXrank (radgraph + green_score envs)
echo ""
echo "=== Step 5/6: ReXrank scoring ==="
echo "[$(date +%H:%M:%S)] Running ReXrank..."
bash scripts/score_rexrank.sh $OUT

# 6. Compare
echo ""
echo "=== Step 6/6: Compare ==="
echo "[$(date +%H:%M:%S)] Comparing..."
$PY scripts/eval_mimic.py --mode compare --output $OUT

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo "=========================================="
