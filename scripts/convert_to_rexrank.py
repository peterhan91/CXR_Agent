#!/usr/bin/env python3
"""
Convert CXR Agent eval results to ReXrank-metric CSV format.

Reads our test_set.json + predictions_*.json and produces:
  - gt_reports_{model}.csv
  - predicted_reports_{model}.csv

in the format expected by ReXrank-metric scripts.

Usage:
    python scripts/convert_to_rexrank.py \
        --eval-dir results/eval_2 \
        --models agent chexone sonnet \
        --rexrank-dir /home/than/DeepLearning/ReXrank-metric \
        --dataset mimic-cxr \
        --split reports
"""

import argparse
import json
import os
import re

import pandas as pd


def clean_report(text):
    """Strip markdown formatting and extra whitespace from agent reports."""
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s+', '', text)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    # Remove leading preamble (agent sometimes starts with "I now have sufficient...")
    preamble_patterns = [
        r'^I now have sufficient.*?(?=FINDINGS|Findings|---|\n\n)',
        r'^Let me compile.*?(?=FINDINGS|Findings|---|\n\n)',
        r'^Based on.*?(?=FINDINGS|Findings|---|\n\n)',
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_sections(report_text):
    """Extract FINDINGS and IMPRESSION sections from a report."""
    findings = ""
    impression = ""

    # Try to find FINDINGS section
    findings_match = re.search(
        r'(?:FINDINGS|Findings)[:\s]*\n?(.*?)(?=IMPRESSION|Impression|$)',
        report_text, re.DOTALL | re.IGNORECASE
    )
    if findings_match:
        findings = findings_match.group(1).strip()

    # Try to find IMPRESSION section
    impression_match = re.search(
        r'(?:IMPRESSION|Impression)[:\s]*\n?(.*?)$',
        report_text, re.DOTALL | re.IGNORECASE
    )
    if impression_match:
        impression = impression_match.group(1).strip()

    return findings, impression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["agent", "chexone"])
    parser.add_argument("--rexrank-dir", default="/home/than/DeepLearning/ReXrank-metric")
    parser.add_argument("--dataset", default="mimic-cxr")
    parser.add_argument("--split", default="reports", choices=["findings", "reports"])
    args = parser.parse_args()

    # Load test set (ground truth)
    test_set_path = os.path.join(args.eval_dir, "test_set.json")
    with open(test_set_path) as f:
        test_set = json.load(f)

    gt_by_study = {str(s["study_id"]): s["report_gt"] for s in test_set}

    for model_name in args.models:
        pred_path = os.path.join(args.eval_dir, f"predictions_{model_name}.json")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {pred_path} not found")
            continue

        with open(pred_path) as f:
            predictions = json.load(f)

        rows_gt = []
        rows_pred = []

        for idx, pred in enumerate(predictions):
            study_id = str(pred["study_id"])
            gt_report = gt_by_study.get(study_id, "")
            pred_report = clean_report(pred["report_pred"])

            if args.split == "findings":
                gt_findings, _ = extract_sections(gt_report)
                pred_findings, _ = extract_sections(pred_report)
                gt_text = gt_findings if gt_findings else gt_report
                pred_text = pred_findings if pred_findings else pred_report
            else:
                gt_text = gt_report
                pred_text = pred_report

            case_id = f"s{study_id}"
            rows_gt.append({"study_id": idx, "case_id": case_id, "report": gt_text})
            rows_pred.append({"study_id": idx, "case_id": case_id, "report": pred_text})

        # Write CSVs
        split_dir = "findings" if args.split == "findings" else "reports"
        out_dir = os.path.join(args.rexrank_dir, "data", split_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)

        gt_csv = os.path.join(out_dir, f"gt_reports_{model_name}.csv")
        pred_csv = os.path.join(out_dir, f"predicted_reports_{model_name}.csv")

        pd.DataFrame(rows_gt).to_csv(gt_csv, index=False)
        pd.DataFrame(rows_pred).to_csv(pred_csv, index=False)
        print(f"[OK] {model_name}: {len(rows_gt)} samples -> {gt_csv}")
        print(f"[OK] {model_name}: {len(rows_pred)} samples -> {pred_csv}")


if __name__ == "__main__":
    main()
