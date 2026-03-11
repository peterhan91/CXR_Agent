#!/usr/bin/env python3
"""
MIMIC-CXR Evaluation: CheXOne baseline vs CXR Agent.

Five modes run in sequence — each builds on the previous:

    # 1. Prepare test set from MIMIC-CXR-JPG metadata
    python scripts/eval_mimic.py --mode prepare \
        --mimic_dir /path/to/mimic-cxr-jpg \
        --reports_dir /path/to/mimic-cxr/files \
        --output results/eval/

    # 2. Run CheXOne baseline (direct server call, no agent, no CLEAR)
    python scripts/eval_mimic.py --mode chexone --output results/eval/

    # 3. Run CXR Agent (full pipeline: CLEAR + tools + ReAct)
    python scripts/eval_mimic.py --mode agent --output results/eval/

    # 4. Score predictions with CXR-Report-Metric (or fallback metrics)
    python scripts/eval_mimic.py --mode score --output results/eval/

    # 5. Compare all scored results side-by-side
    python scripts/eval_mimic.py --mode compare --output results/eval/

Output directory structure:
    results/eval/
    ├── test_set.json              # Prepared test set (study_id, image_path, report_gt)
    ├── predictions_chexone.json   # CheXOne baseline predictions
    ├── predictions_agent.json     # CXR Agent predictions
    ├── gt_chexone.csv             # GT reports in CXR-Report-Metric format
    ├── pred_chexone.csv           # Predicted reports in CXR-Report-Metric format
    ├── scores_chexone.json        # Metric scores for CheXOne
    ├── scores_agent.json          # Metric scores for Agent
    └── comparison.txt             # Side-by-side metric table
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ─── Report Parsing ─────────────────────────────────────────────────────────


def parse_report_sections(text: str) -> tuple:
    """Extract FINDINGS and IMPRESSION from raw MIMIC-CXR report text.

    Handles common format variations:
    - FINDINGS: / FINDING: with or without newline after colon
    - IMPRESSION: / CONCLUSION: as impression header
    - Reports with only one section present
    - Free-text reports with no section headers (returns empty strings)

    Returns:
        (findings_text, impression_text) — either may be empty string
    """
    findings = ""
    impression = ""

    # FINDINGS section: capture until next section header or end of text
    m = re.search(
        r"FINDINGS?:?\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|RECOMMENDATION)\s*:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        findings = m.group(1).strip()

    # IMPRESSION section (also matches CONCLUSION)
    m = re.search(
        r"(?:IMPRESSION|CONCLUSION):?\s*(.*?)(?=(?:RECOMMENDATION|NOTIFICATION|ADDENDUM)\s*:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        impression = m.group(1).strip()

    return findings, impression


def format_gt_report(findings: str, impression: str, raw_text: str) -> str:
    """Format ground truth report for evaluation.

    Prefers structured FINDINGS + IMPRESSION. Falls back to raw text
    if no sections were parsed.
    """
    parts = []
    if findings:
        parts.append(f"FINDINGS:\n{findings}")
    if impression:
        parts.append(f"IMPRESSION:\n{impression}")

    if parts:
        return "\n\n".join(parts)

    # Fallback: use raw text if no sections found
    return raw_text.strip()


# ─── Test Set Preparation ───────────────────────────────────────────────────


def prepare_test_set(args):
    """Build test_set.json from MIMIC-CXR-JPG metadata.

    Reads the official MIMIC-CXR split and metadata CSVs, selects frontal
    images (PA preferred over AP) for test-split studies, loads ground truth
    report text, and saves a JSON file for subsequent eval runs.

    Expected directory layout:
        mimic_dir (mimic-cxr-jpg/):
            mimic-cxr-2.0.0-split.csv[.gz]
            mimic-cxr-2.0.0-metadata.csv[.gz]
            files/p10/p10000032/s50414267/02aa804e-....jpg

        reports_dir (mimic-cxr/files/ or mimic-cxr-jpg/ if reports co-located):
            p10/p10000032/s50414267.txt
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for prepare mode: pip install pandas")
        sys.exit(1)

    mimic_dir = Path(args.mimic_dir)
    reports_dir = Path(args.reports_dir) if args.reports_dir else mimic_dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect files/ subdirectory for reports
    if (reports_dir / "files").is_dir():
        reports_base = reports_dir / "files"
    else:
        reports_base = reports_dir

    # Load split CSV
    split_path = _find_csv(mimic_dir, "mimic-cxr-2.0.0-split")
    if not split_path:
        logger.error(f"Split CSV not found in {mimic_dir}")
        sys.exit(1)

    logger.info(f"Loading split from {split_path}")
    split_df = pd.read_csv(split_path)
    test_df = split_df[split_df["split"] == "test"].copy()
    logger.info(f"Test split: {len(test_df)} images")

    # Load metadata CSV
    meta_path = _find_csv(mimic_dir, "mimic-cxr-2.0.0-metadata")
    if not meta_path:
        logger.error(f"Metadata CSV not found in {mimic_dir}")
        sys.exit(1)

    logger.info(f"Loading metadata from {meta_path}")
    meta_df = pd.read_csv(meta_path)

    # Merge test split with metadata to get ViewPosition
    test_meta = test_df.merge(
        meta_df[["dicom_id", "subject_id", "study_id", "ViewPosition"]],
        on=["dicom_id", "subject_id", "study_id"],
        how="left",
    )

    # Filter for frontal views only
    frontal = test_meta[test_meta["ViewPosition"].isin(["PA", "AP"])].copy()
    logger.info(f"Frontal test images: {len(frontal)}")

    # Pick best frontal per study (PA preferred over AP)
    frontal["_priority"] = frontal["ViewPosition"].map({"PA": 0, "AP": 1})
    frontal = frontal.sort_values("_priority").drop_duplicates("study_id", keep="first")
    logger.info(f"Unique test studies with frontal view: {len(frontal)}")

    if args.max_samples:
        frontal = frontal.head(args.max_samples)
        logger.info(f"Limited to {args.max_samples} samples")

    # Build test set entries
    test_set = []
    skipped = {"no_image": 0, "no_report": 0, "empty_report": 0}

    for _, row in frontal.iterrows():
        sid = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        did = str(row["dicom_id"])
        prefix = f"p{sid[:2]}"

        img_path = mimic_dir / "files" / prefix / f"p{sid}" / f"s{stid}" / f"{did}.jpg"
        rpt_path = reports_base / prefix / f"p{sid}" / f"s{stid}.txt"

        if not img_path.exists():
            skipped["no_image"] += 1
            continue
        if not rpt_path.exists():
            skipped["no_report"] += 1
            continue

        raw = rpt_path.read_text()
        findings, impression = parse_report_sections(raw)
        gt = format_gt_report(findings, impression, raw)

        if not gt.strip():
            skipped["empty_report"] += 1
            continue

        test_set.append({
            "study_id": stid,
            "subject_id": sid,
            "dicom_id": did,
            "image_path": str(img_path),
            "report_gt": gt,
        })

    out_path = output_dir / "test_set.json"
    with open(out_path, "w") as f:
        json.dump(test_set, f, indent=2)

    logger.info(f"Test set: {len(test_set)} studies -> {out_path}")
    logger.info(f"Skipped: {skipped}")
    print(f"\nPrepared: {len(test_set)} studies -> {out_path}")
    print(f"Skipped: {skipped}")


def _find_csv(directory: Path, stem: str) -> Path:
    """Find a CSV file by stem, trying .csv.gz then .csv."""
    for ext in [".csv.gz", ".csv"]:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    return None


# ─── CheXOne Baseline ───────────────────────────────────────────────────────


def run_chexone(args):
    """Generate reports using CheXOne server (baseline, no agent, no CLEAR).

    Calls the CheXOne FastAPI server directly for each image.
    Saves results incrementally for resume support.
    """
    import requests as req_lib

    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir)

    endpoint = args.chexone_endpoint
    predictions_path = output_dir / "predictions_chexone.json"

    # Resume: load existing predictions
    existing = _load_existing_predictions(predictions_path)
    predictions = list(existing.values())
    total = len(test_set)
    errors = 0
    t_start = time.time()

    for i, entry in enumerate(test_set):
        study_id = entry["study_id"]
        if study_id in existing:
            continue

        t0 = time.time()
        try:
            resp = req_lib.post(
                f"{endpoint}/generate_report",
                json={"image_path": entry["image_path"], "reasoning": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            report = data["report"]
            gen_time = data.get("generation_time_ms", 0)
        except Exception as e:
            logger.error(f"[{i+1}/{total}] Failed study {study_id}: {e}")
            report = ""
            gen_time = 0
            errors += 1

        wall_ms = (time.time() - t0) * 1000

        pred = {
            "study_id": study_id,
            "report_pred": report,
            "generation_time_ms": gen_time,
            "wall_time_ms": wall_ms,
        }
        predictions.append(pred)
        existing[study_id] = pred

        # Save every 10 studies
        if len(predictions) % 10 == 0:
            _save_predictions(predictions_path, predictions)
            done = len(predictions)
            elapsed = time.time() - t_start
            rate = done / elapsed * 3600 if elapsed > 0 else 0
            logger.info(f"[{done}/{total}] saved ({rate:.0f} studies/hr)")

    _save_predictions(predictions_path, predictions)
    _print_summary("CheXOne", predictions, errors, predictions_path)


# ─── CXR Agent ───────────────────────────────────────────────────────────────


def run_agent_eval(args):
    """Generate reports using the full CXR Agent pipeline.

    Full pipeline: CLEAR concept scoring + 14 tools + ReAct agent.
    Saves results incrementally with token usage tracking.
    """
    import yaml
    from agent.react_agent import CXRReActAgent

    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    predictions_path = output_dir / "predictions_agent.json"

    # Resume
    existing = _load_existing_predictions(predictions_path)
    predictions = list(existing.values())

    # CLEAR scorer
    scorer = None
    if not args.no_clear:
        from clear.concept_scorer import CLEARConceptScorer

        clear_cfg = config.get("clear", {})
        scorer = CLEARConceptScorer(
            model_path=clear_cfg.get("model_path"),
            concepts_path=clear_cfg.get("concepts_path"),
            dinov2_model_name=clear_cfg.get("dinov2_model_name", "dinov2_vitb14"),
            image_resolution=clear_cfg.get("image_resolution", 448),
        )
        logger.info("Loading CLEAR model...")
        scorer.load()
        logger.info("CLEAR model ready")

    # Build tools
    tools = _build_tools(config)

    # Initialize agent
    acfg = config.get("agent", {})
    agent = CXRReActAgent(
        model=acfg.get("model", "claude-sonnet-4-6"),
        max_iterations=acfg.get("max_iterations", 10),
        max_tokens=acfg.get("max_tokens", 4096),
        temperature=acfg.get("temperature", 0.0),
        tools=tools,
        reasoning_effort=acfg.get("reasoning_effort"),
    )

    total = len(test_set)
    cum_in = sum(p.get("input_tokens", 0) for p in predictions)
    cum_out = sum(p.get("output_tokens", 0) for p in predictions)
    errors = 0
    t_start = time.time()

    for i, entry in enumerate(test_set):
        study_id = entry["study_id"]
        if study_id in existing:
            continue

        logger.info(f"[{i+1}/{total}] Agent: study {study_id}")
        t0 = time.time()

        # CLEAR concept prior
        concept_prior = ""
        if scorer:
            top_k = config.get("clear", {}).get("top_k", 20)
            concept_prior = scorer.score_image(entry["image_path"], top_k=top_k)

        try:
            trajectory = agent.run(
                image_path=entry["image_path"],
                concept_prior_text=concept_prior,
                image_id=study_id,
            )
            report = trajectory.final_report
            in_tok = trajectory.total_input_tokens
            out_tok = trajectory.total_output_tokens
            n_steps = len(trajectory.steps)
        except Exception as e:
            logger.error(f"Agent failed for {study_id}: {e}", exc_info=True)
            report = ""
            in_tok = out_tok = n_steps = 0
            errors += 1

        wall_ms = (time.time() - t0) * 1000
        cum_in += in_tok
        cum_out += out_tok

        pred = {
            "study_id": study_id,
            "report_pred": report,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "num_steps": n_steps,
            "wall_time_ms": wall_ms,
        }
        predictions.append(pred)
        existing[study_id] = pred

        # Save every 5 studies
        if len(predictions) % 5 == 0:
            _save_predictions(predictions_path, predictions)
            done = len(predictions)
            elapsed = time.time() - t_start
            logger.info(
                f"[{done}/{total}] saved | "
                f"tokens: {cum_in:,} in / {cum_out:,} out | "
                f"{elapsed/60:.1f}min elapsed"
            )

    _save_predictions(predictions_path, predictions)

    print(f"\nAgent: {len(predictions)} predictions -> {predictions_path}")
    print(f"Total tokens: {cum_in:,} input, {cum_out:,} output")
    if errors:
        print(f"Errors: {errors}")


# ─── Scoring ─────────────────────────────────────────────────────────────────


def score_predictions(args):
    """Score predictions using CXR-Report-Metric (or fallback metrics).

    Looks for all predictions_*.json files in the output directory,
    exports GT/pred CSVs in CXR-Report-Metric format, and computes scores.

    CXR-Report-Metric (https://github.com/rajpurkarlab/CXR-Report-Metric):
        pip install CXR-Report-Metric
        Requires: CheXbert checkpoint, RadGraph PhysioNet access
        Metrics: RadCliQ-v1, RadGraph-F1, SembScore, BERTScore, BLEU-2

    Fallback: BLEU-1, BLEU-2, average report lengths
    """
    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir)
    gt_by_study = {e["study_id"]: e["report_gt"] for e in test_set}

    # Find all prediction files
    pred_files = sorted(output_dir.glob("predictions_*.json"))
    if not pred_files:
        logger.error("No prediction files found. Run --mode chexone or --mode agent first.")
        sys.exit(1)

    for pred_path in pred_files:
        mode_name = pred_path.stem.replace("predictions_", "")
        logger.info(f"Scoring: {mode_name}")

        with open(pred_path) as f:
            preds = json.load(f)

        # Build aligned GT/pred lists (skip empty predictions)
        gt_reports = []
        pred_reports = []
        study_ids = []

        for p in preds:
            sid = p["study_id"]
            if sid in gt_by_study and p.get("report_pred", "").strip():
                study_ids.append(sid)
                gt_reports.append(gt_by_study[sid])
                pred_reports.append(p["report_pred"])

        if not study_ids:
            logger.warning(f"No valid predictions for {mode_name}, skipping")
            continue

        logger.info(f"  {len(study_ids)} studies with valid predictions")

        # Export CSVs for CXR-Report-Metric
        gt_csv = output_dir / f"gt_{mode_name}.csv"
        pred_csv = output_dir / f"pred_{mode_name}.csv"
        _write_report_csv(gt_csv, study_ids, gt_reports)
        _write_report_csv(pred_csv, study_ids, pred_reports)

        # Try CXR-Report-Metric first, fall back to basic metrics
        scores_path = output_dir / f"scores_{mode_name}.json"
        try:
            from CXRMetric.run_eval import calc_metric

            logger.info("  Using CXR-Report-Metric (RadCliQ-v1, RadGraph-F1, ...)")
            calc_metric(str(gt_csv), str(pred_csv), str(scores_path), use_idf=True)
            logger.info(f"  Scores saved: {scores_path}")
        except ImportError:
            logger.warning(
                "  CXR-Report-Metric not installed. Using basic metrics.\n"
                "  Install: pip install CXR-Report-Metric\n"
                "  See: https://github.com/rajpurkarlab/CXR-Report-Metric"
            )
            _compute_basic_metrics(gt_reports, pred_reports, scores_path)
        except Exception as e:
            logger.error(f"  CXR-Report-Metric error: {e}")
            logger.info("  Falling back to basic metrics")
            _compute_basic_metrics(gt_reports, pred_reports, scores_path)

    print(f"\nScoring complete. Results in {output_dir}")


def _write_report_csv(path: Path, study_ids: list, reports: list):
    """Write report CSV in CXR-Report-Metric format (study_id, report)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "report"])
        for sid, report in zip(study_ids, reports):
            writer.writerow([sid, report])


def _compute_basic_metrics(gt_reports: list, pred_reports: list, output_path: Path):
    """Compute basic text metrics as fallback when CXR-Report-Metric unavailable."""
    n = len(gt_reports)
    avg_gt_len = sum(len(r.split()) for r in gt_reports) / n
    avg_pred_len = sum(len(p.split()) for p in pred_reports) / n

    scores = {
        "num_studies": n,
        "avg_gt_length_words": round(avg_gt_len, 1),
        "avg_pred_length_words": round(avg_pred_len, 1),
    }

    # BLEU scores (if nltk available)
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        refs = [[r.split()] for r in gt_reports]
        hyps = [p.split() for p in pred_reports]
        smooth = SmoothingFunction().method1
        scores["bleu_1"] = round(
            corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth), 4
        )
        scores["bleu_2"] = round(
            corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth), 4
        )
    except ImportError:
        logger.warning("  nltk not installed, skipping BLEU (pip install nltk)")

    # BERTScore (if available)
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(pred_reports, gt_reports, lang="en", verbose=False)
        scores["bertscore_f1"] = round(F1.mean().item(), 4)
    except ImportError:
        logger.warning("  bert-score not installed, skipping BERTScore (pip install bert-score)")

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    logger.info(f"  Basic metrics: {scores}")


# ─── Compare ─────────────────────────────────────────────────────────────────


def compare_results(args):
    """Print side-by-side comparison of all scored results."""
    output_dir = Path(args.output)

    score_files = sorted(output_dir.glob("scores_*.json"))
    if not score_files:
        logger.error("No score files found. Run --mode score first.")
        sys.exit(1)

    results = {}
    for sf in score_files:
        mode = sf.stem.replace("scores_", "")
        with open(sf) as f:
            results[mode] = json.load(f)

    # Collect all metric keys
    all_keys = []
    seen = set()
    for scores in results.values():
        for k in scores:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build table
    modes = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>18}" for m in modes)
    separator = "-" * len(header)

    lines = []
    lines.append("=" * len(header))
    lines.append("MIMIC-CXR Evaluation Results")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(separator)

    for key in all_keys:
        row = f"{key:<30}"
        for mode in modes:
            val = results[mode].get(key, "—")
            if isinstance(val, float):
                row += f"{val:>18.4f}"
            else:
                row += f"{str(val):>18}"
        lines.append(row)

    lines.append("=" * len(header))

    table = "\n".join(lines)
    print(f"\n{table}")

    # Save
    compare_path = output_dir / "comparison.txt"
    with open(compare_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved to {compare_path}")


# ─── Shared Utilities ────────────────────────────────────────────────────────


def _load_test_set(output_dir: Path) -> list:
    """Load test_set.json, exit with error if not found."""
    test_set_path = output_dir / "test_set.json"
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        logger.error("Run --mode prepare first to build the test set.")
        sys.exit(1)
    with open(test_set_path) as f:
        test_set = json.load(f)
    logger.info(f"Loaded test set: {len(test_set)} studies")
    return test_set


def _load_existing_predictions(path: Path) -> dict:
    """Load existing predictions for resume support."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for entry in json.load(f):
                existing[entry["study_id"]] = entry
        logger.info(f"Resuming from {len(existing)} existing predictions")
    return existing


def _save_predictions(path: Path, predictions: list):
    """Save predictions JSON atomically."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(predictions, f, indent=2)
    tmp_path.rename(path)


def _print_summary(mode_name: str, predictions: list, errors: int, path: Path):
    """Print summary statistics for a prediction run."""
    n = len(predictions)
    non_empty = sum(1 for p in predictions if p.get("report_pred", "").strip())
    total_wall = sum(p.get("wall_time_ms", 0) for p in predictions)
    avg_wall = total_wall / n if n else 0

    print(f"\n{mode_name}: {n} predictions -> {path}")
    print(f"  Non-empty: {non_empty}/{n}")
    print(f"  Avg wall time: {avg_wall:.0f}ms")
    if errors:
        print(f"  Errors: {errors}")


def _build_tools(config: dict) -> list:
    """Build tool instances from config (mirrors run_agent.py:build_tools)."""
    from tools import (
        CheXagent2ReportTool,
        CheXagent2SRRGTool,
        CheXagent2GroundingTool,
        CheXagent2ClassifyTool,
        CheXagent2VQATool,
        CheXOneReportTool,
        MedVersaReportTool,
        MedVersaClassifyTool,
        MedVersaDetectTool,
        MedVersaSegmentTool,
        MedVersaVQATool,
        BiomedParseSegmentTool,
        MedSAM3SegmentTool,
        FactCheXckerVerifyTool,
    )

    tool_config = config.get("tools", {})
    registry = {
        "chexagent2": CheXagent2ReportTool,
        "chexagent2_srrg": CheXagent2SRRGTool,
        "chexagent2_classify": CheXagent2ClassifyTool,
        "chexagent2_grounding": CheXagent2GroundingTool,
        "chexagent2_vqa": CheXagent2VQATool,
        "chexone": CheXOneReportTool,
        "medversa": MedVersaReportTool,
        "medversa_classify": MedVersaClassifyTool,
        "medversa_detect": MedVersaDetectTool,
        "medversa_segment": MedVersaSegmentTool,
        "medversa_vqa": MedVersaVQATool,
        "biomedparse": BiomedParseSegmentTool,
        "medsam3": MedSAM3SegmentTool,
        "factchexcker": FactCheXckerVerifyTool,
    }

    tools = []
    for key, cls in registry.items():
        entry = tool_config.get(key, {})
        if entry.get("enabled", False):
            endpoint = entry.get("endpoint", "http://localhost:8000")
            tools.append(cls(endpoint=endpoint))
            logger.info(f"  Tool enabled: {key} -> {endpoint}")

    logger.info(f"Built {len(tools)} tools")
    return tools


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-CXR Evaluation: CheXOne baseline vs CXR Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (run each step in order):
  python scripts/eval_mimic.py --mode prepare --mimic_dir /data/mimic-cxr-jpg --reports_dir /data/mimic-cxr/files --output results/eval/
  python scripts/eval_mimic.py --mode chexone --output results/eval/
  python scripts/eval_mimic.py --mode agent --output results/eval/
  python scripts/eval_mimic.py --mode score --output results/eval/
  python scripts/eval_mimic.py --mode compare --output results/eval/

  # Quick test with 50 samples:
  python scripts/eval_mimic.py --mode prepare --mimic_dir /data/mimic-cxr-jpg --reports_dir /data/mimic-cxr/files --output results/eval/ --max_samples 50
        """,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["prepare", "chexone", "agent", "score", "compare"],
        help="prepare: build test set | chexone: baseline | agent: full pipeline | score: compute metrics | compare: print table",
    )
    parser.add_argument(
        "--output",
        default="results/eval/",
        help="Output directory for all results (default: results/eval/)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Agent config YAML (for agent mode)",
    )

    # Prepare mode
    parser.add_argument("--mimic_dir", help="MIMIC-CXR-JPG root directory")
    parser.add_argument(
        "--reports_dir",
        help="Directory with report .txt files (default: same as mimic_dir)",
    )
    parser.add_argument("--max_samples", type=int, help="Limit test set size (for debugging)")

    # CheXOne mode
    parser.add_argument(
        "--chexone_endpoint",
        default="http://localhost:8002",
        help="CheXOne server URL (default: http://localhost:8002)",
    )

    # Agent mode
    parser.add_argument("--no_clear", action="store_true", help="Skip CLEAR concept scoring")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "prepare" and not args.mimic_dir:
        parser.error("--mimic_dir required for prepare mode")

    dispatch = {
        "prepare": prepare_test_set,
        "chexone": run_chexone,
        "agent": run_agent_eval,
        "score": score_predictions,
        "compare": compare_results,
    }

    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
