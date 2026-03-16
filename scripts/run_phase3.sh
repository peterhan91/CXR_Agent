#!/usr/bin/env bash
# Phase 3: Run baselines + agent on 5-study samples for all 4 datasets
set -e

OUT=results/eval_v51
CFG=configs/config_grounded.yaml

for DATASET in mimic_cxr chexpert_plus rexgradient iu_xray; do
  INPUT=data/eval/sample_5/${DATASET}_5.json
  [ -f "$INPUT" ] || { echo "SKIP $DATASET: $INPUT missing"; continue; }

  echo ""
  echo "================================================================"
  echo "DATASET: $DATASET"
  echo "================================================================"

  # CheXOne baseline
  echo "--- CheXOne baseline ---"
  python scripts/eval_mimic.py --mode chexone \
    --input "$INPUT" --output "$OUT/$DATASET/" 2>&1 | tail -5

  # CheXagent-2 baseline
  echo "--- CheXagent-2 baseline ---"
  python scripts/eval_mimic.py --mode chexagent2 \
    --input "$INPUT" --output "$OUT/$DATASET/" 2>&1 | tail -5

  # MedGemma baseline
  echo "--- MedGemma baseline ---"
  python scripts/eval_mimic.py --mode medgemma \
    --input "$INPUT" --output "$OUT/$DATASET/" 2>&1 | tail -5

  # MedVersa baseline
  echo "--- MedVersa baseline ---"
  python scripts/eval_mimic.py --mode medversa \
    --input "$INPUT" --output "$OUT/$DATASET/" 2>&1 | tail -5

  # Agent (plain prompt, no skills)
  echo "--- Agent (no_skills) ---"
  python scripts/eval_mimic.py --mode agent \
    --input "$INPUT" --output "$OUT/$DATASET/" \
    --config $CFG --no_skills 2>&1 | tail -5

  echo "DONE: $DATASET"
done

echo ""
echo "================================================================"
echo "All predictions complete. Starting scoring..."
echo "================================================================"
