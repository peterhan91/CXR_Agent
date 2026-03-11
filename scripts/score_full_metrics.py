#!/usr/bin/env python3
"""
Score CXR reports with full CXR-Report-Metric (all 5 ReXrank metrics).

Must be run from the CXR-Report-Metric directory in the `radgraph` conda env:

    cd /path/to/ReXrank-metric/scripts/CXR-Report-Metric && \
    conda run -n radgraph python /path/to/CXR_Agent/scripts/score_full_metrics.py \
        --output /path/to/CXR_Agent/results/eval/

This reads gt_*.csv and pred_*.csv files produced by eval_mimic.py --mode score
(or generates them from test_set.json + predictions_*.json), runs CXR-Report-Metric,
and writes scores_*.json in the format eval_mimic.py --mode compare expects.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# CXR-Report-Metric must be importable from cwd
sys.path.insert(0, os.getcwd())
from CXRMetric.run_eval import calc_metric, CompositeMetric  # noqa: F401


def write_report_csv(path, study_ids, reports):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "report"])
        for sid, report in zip(study_ids, reports):
            writer.writerow([sid, report])


def csv_to_scores_json(csv_path, json_path):
    """Convert CXR-Report-Metric output CSV to scores JSON for eval_mimic.py."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    scores = {"num_studies": len(df)}

    col_map = {
        "bleu_score": "bleu_2",
        "bertscore": "bertscore_f1",
        "semb_score": "semb_score",
        "radgraph_combined": "radgraph_f1",
        "RadCliQ-v1": "radcliq_v1",
        "RadCliQ-v0": "radcliq_v0",
    }

    for csv_col, json_key in col_map.items():
        if csv_col in df.columns:
            scores[json_key] = round(float(df[csv_col].mean()), 4)

    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"  Scores: {json.dumps(scores, indent=2)}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="eval_mimic.py output directory")
    parser.add_argument("--use-idf", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    test_set_path = output_dir / "test_set.json"

    if not test_set_path.exists():
        print(f"ERROR: test_set.json not found in {output_dir}")
        sys.exit(1)

    with open(test_set_path) as f:
        test_set = json.load(f)
    gt_by_study = {e["study_id"]: e["report_gt"] for e in test_set}

    # Find all prediction files
    pred_files = sorted(output_dir.glob("predictions_*.json"))
    if not pred_files:
        print("ERROR: No prediction files found")
        sys.exit(1)

    for pred_path in pred_files:
        mode_name = pred_path.stem.replace("predictions_", "")
        print(f"\n=== Scoring: {mode_name} ===")

        with open(pred_path) as f:
            preds = json.load(f)

        # Build aligned GT/pred lists
        gt_reports, pred_reports, study_ids = [], [], []
        for p in preds:
            sid = p["study_id"]
            if sid in gt_by_study and p.get("report_pred", "").strip():
                study_ids.append(sid)
                gt_reports.append(gt_by_study[sid])
                pred_reports.append(p["report_pred"])

        if not study_ids:
            print(f"  No valid predictions for {mode_name}, skipping")
            continue

        print(f"  {len(study_ids)} studies")

        # Write CSVs
        gt_csv = output_dir / f"gt_{mode_name}.csv"
        pred_csv = output_dir / f"pred_{mode_name}.csv"
        write_report_csv(gt_csv, study_ids, gt_reports)
        write_report_csv(pred_csv, study_ids, pred_reports)

        # Run CXR-Report-Metric (outputs CSV)
        out_csv = output_dir / f"report_scores_{mode_name}.csv"
        print(f"  Running CXR-Report-Metric...")
        calc_metric(str(gt_csv), str(pred_csv), str(out_csv), args.use_idf)

        # Convert to JSON for eval_mimic.py compare mode
        scores_json = output_dir / f"scores_{mode_name}.json"
        csv_to_scores_json(str(out_csv), str(scores_json))
        print(f"  Saved: {scores_json}")

    print("\nDone! Run: python scripts/eval_mimic.py --mode compare --output", str(output_dir))


if __name__ == "__main__":
    main()
