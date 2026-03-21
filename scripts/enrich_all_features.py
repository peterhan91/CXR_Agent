#!/usr/bin/env python3
"""Enrich all 4 eval datasets with missing feature fields in-place.

Adds/updates:
- lateral_image_path: merged from data/eval/enriched/ (pre-computed by enrich_laterals.py)
- metadata.age / metadata.sex: promoted from admission_info.demographics (MIMIC only)
- metadata.indication: extracted from report_gt preamble (MIMIC only)

IU, CheXpert+, RexGradient already have age/sex/indication in metadata — kept as-is.

Usage:
    python scripts/enrich_all_features.py [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
ENRICHED_DIR = DATA_DIR / "enriched"
BACKUP_DIR = DATA_DIR / "backup_pre_enrich"

DATASETS = [
    "mimic_cxr_test.json",
    "iu_xray_test.json",
    "chexpert_plus_valid.json",
    "rexgradient_test.json",
]


def extract_indication_from_report(report_gt: str) -> str:
    """Extract INDICATION/HISTORY/CLINICAL INFORMATION from report preamble.

    Only extracts text before FINDINGS/IMPRESSION — never touches diagnostic content.
    """
    if not report_gt:
        return ""

    # Match INDICATION, CLINICAL INFORMATION, CLINICAL HISTORY, HISTORY, REASON FOR EXAM
    m = re.search(
        r"(?:INDICATION|CLINICAL\s+(?:INFORMATION|HISTORY)|HISTORY|REASON\s+FOR\s+(?:EXAM|STUDY))\s*:?\s*"
        r"(.*?)(?=\n\s*(?:COMPARISON|TECHNIQUE|FINDINGS|IMPRESSION|CONCLUSION|RECOMMENDATION|"
        r"PA\s+AND\s+LATERAL|FRONTAL|SINGLE|CHEST)\s*:|\Z)",
        report_gt,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip()
        # Clean up de-identified placeholders
        text = re.sub(r"\s+", " ", text)
        if text and text != "___" and text != "___.":
            return text
    return ""


def enrich_mimic(data: list[dict], enriched_laterals: dict[str, str]) -> dict:
    """Enrich MIMIC-CXR with laterals, age/sex promotion, and indication extraction."""
    stats = {"lateral": 0, "age": 0, "sex": 0, "indication": 0}

    for entry in data:
        sid = entry["study_id"]
        meta = entry.get("metadata", {})

        # 1. Lateral — merge from enriched
        lat = enriched_laterals.get(sid)
        if lat:
            entry["lateral_image_path"] = lat
            stats["lateral"] += 1
        elif "lateral_image_path" not in entry:
            entry["lateral_image_path"] = None

        # 2. Age — promote from admission_info.demographics
        if not meta.get("age"):
            adm_age = (
                meta.get("admission_info", {})
                .get("demographics", {})
                .get("age")
            )
            if adm_age is not None:
                meta["age"] = adm_age
                stats["age"] += 1

        # 3. Sex — promote from admission_info.demographics
        if not meta.get("sex"):
            adm_gender = (
                meta.get("admission_info", {})
                .get("demographics", {})
                .get("gender")
            )
            if adm_gender:
                meta["sex"] = adm_gender
                stats["sex"] += 1

        # 4. Indication — extract from report_gt preamble
        if not (meta.get("indication") or "").strip():
            indication = extract_indication_from_report(entry.get("report_gt", ""))
            if indication:
                meta["indication"] = indication
                stats["indication"] += 1

        entry["metadata"] = meta

    return stats


def enrich_other(data: list[dict], enriched_laterals: dict[str, str]) -> dict:
    """Enrich IU/CheXpert+/RexGradient — only add laterals (rest already in metadata)."""
    stats = {"lateral": 0}

    for entry in data:
        sid = entry["study_id"]
        lat = enriched_laterals.get(sid)
        if lat:
            entry["lateral_image_path"] = lat
            stats["lateral"] += 1
        elif "lateral_image_path" not in entry:
            entry["lateral_image_path"] = None

    return stats


def load_enriched_laterals(fname: str) -> dict[str, str]:
    """Load lateral_image_path mapping from enriched dataset."""
    path = ENRICHED_DIR / fname
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping lateral enrichment")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        s["study_id"]: s["lateral_image_path"]
        for s in data
        if s.get("lateral_image_path")
    }


def audit_dataset(data: list[dict], label: str):
    """Print feature coverage for a dataset."""
    n = len(data)
    lat = sum(1 for s in data if s.get("lateral_image_path"))
    prior = sum(
        1
        for s in data
        if s.get("prior_study") and s["prior_study"].get("image_path")
    )
    age = sum(1 for s in data if s.get("metadata", {}).get("age"))
    sex = sum(1 for s in data if s.get("metadata", {}).get("sex"))
    ind = sum(
        1
        for s in data
        if (s.get("metadata", {}).get("indication") or "").strip()
    )
    comp = sum(
        1
        for s in data
        if (s.get("metadata", {}).get("comparison") or "").strip()
    )

    print(f"  {label:<16} n={n:>5}  lateral={lat:>5}  prior={prior:>5}  "
          f"age={age:>5}  sex={sex:>5}  indication={ind:>5}  comparison={comp:>5}")


def main():
    parser = argparse.ArgumentParser(description="Enrich eval datasets with all feature fields")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    # Back up originals
    if not args.dry_run:
        BACKUP_DIR.mkdir(exist_ok=True)
        for fname in DATASETS:
            src = DATA_DIR / fname
            dst = BACKUP_DIR / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"Backed up {fname} -> {BACKUP_DIR}/")

    print("\n=== Enrichment ===\n")

    for fname in DATASETS:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"SKIP: {fname} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        laterals = load_enriched_laterals(fname)
        is_mimic = "mimic" in fname

        if is_mimic:
            stats = enrich_mimic(data, laterals)
            print(f"{fname}: enriched {len(data)} studies")
            print(f"  lateral={stats['lateral']}, age={stats['age']}, "
                  f"sex={stats['sex']}, indication={stats['indication']}")
        else:
            stats = enrich_other(data, laterals)
            print(f"{fname}: enriched {len(data)} studies")
            print(f"  lateral={stats['lateral']} (age/sex/indication already in metadata)")

        if not args.dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  -> Written to {path}")

    # Final audit
    print("\n=== Final Audit ===\n")
    print(f"  {'Dataset':<16} {'n':>5}  {'lateral':>7}  {'prior':>5}  "
          f"{'age':>5}  {'sex':>5}  {'indication':>10}  {'comparison':>10}")
    print("  " + "-" * 85)

    for fname in DATASETS:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        label = fname.replace("_test.json", "").replace("_valid.json", "")
        audit_dataset(data, label)


if __name__ == "__main__":
    main()
