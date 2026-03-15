#!/bin/bash
# Regenerate all grounded images with fixed CheXagent-2 bbox coords.
# Requires the fixed CheXagent-2 server to be running on port 8001.
#
# Usage:
#   nohup bash scripts/regen_grounded_all.sh > logs/regen_grounded.log 2>&1 &

set -e

PY="/home/than/anaconda3/envs/cxr_agent/bin/python"
INPUT="data/eval/sample_100.json"

echo "=== Regen grounded images (bbox aspect ratio fix) ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for track in baseline followup; do
    for variant in "" "_noclear"; do
        dir="results/eval_${track}${variant}"
        pred="$dir/predictions_agent.json"
        if [ ! -f "$pred" ]; then
            echo "[SKIP] $dir — no predictions_agent.json"
            continue
        fi
        echo "=== $dir ==="
        $PY scripts/eval_mimic.py \
            --mode regen-grounded \
            --output "$dir" \
            --input "$INPUT" \
            --track "$track" \
            --chexagent2_endpoint http://localhost:8001
        echo ""
    done
done

echo "=== ALL DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
