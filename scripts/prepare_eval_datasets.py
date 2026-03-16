#!/usr/bin/env python3
"""Prepare standardized evaluation datasets for CXR Agent.

Processes CXR datasets into a unified JSON format, applying
Flamingo-CXR-style filtering (frontal views, FINDINGS section required,
IMPRESSION optional).  Classifies each study as baseline or follow-up
and links prior studies where available.

Default datasets (all require structured FINDINGS + IMPRESSION):
  1. MIMIC-CXR (test)
  2. CheXpert-Plus (valid)
  3. ReXGradient (test)
  4. IU-Xray (test per ReXrank)

Excluded by default:
  - PadChest-GR: translated sentence fragments without FINDINGS/IMPRESSION
    structure; incompatible with report generation metrics. Can still be
    run via --datasets padchest_gr.

Output: data/eval/{dataset}_{split}.json + data/eval/summary.json
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# ──────────────────────────── paths ────────────────────────────

MIMIC_CXR_DIR = "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0"
MIMIC_IV_DIR = "/home/than/physionet.org/files/mimiciv/3.1"
MIMIC_ENRICHED = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "eval_enriched",
)
CHEXPERT_DIR = "/home/than/Datasets/stanford_mit_chest/CheXpert-v1.0"
REXGRADIENT_META = "/home/than/Datasets/RexGradient/data/metadata"
REXGRADIENT_IMAGES = (
    "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K"
    "/snapshots/e0c7dd5940c6e5f77aac20eea3ff93825d7f8ff3/images/deid_png"
)
IU_XRAY_DIR = "/home/than/Datasets/IU_XRay"
REXRANK_DIR = "/home/than/DeepLearning/ReXrank-metric"
PADCHEST_GR_DIR = "/home/than/DeepLearning/Datasets/PadChest-GR"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval")


# ──────────────────────────── helpers ────────────────────────────

def clean_report(text: str) -> str:
    """Remove redundant whitespace, normalize line breaks."""
    if not text:
        return ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(report: str) -> tuple[str, str]:
    """Extract FINDINGS and IMPRESSION from a free-text report."""
    findings = ""
    impression = ""

    # Try structured extraction
    f_match = re.search(
        r"FINDINGS?[:\s]*(.*?)(?=\n\s*(?:IMPRESSION|$))",
        report, re.IGNORECASE | re.DOTALL,
    )
    i_match = re.search(
        r"IMPRESSIONS?[:\s]*(.*?)(?=\n\s*(?:FINDINGS?|RECOMMENDATION|NOTIFICATION|$))",
        report, re.IGNORECASE | re.DOTALL,
    )

    if f_match:
        findings = f_match.group(1).strip()
    if i_match:
        impression = i_match.group(1).strip()

    return findings, impression


def has_prior_reference_comparison(comparison_text: str) -> bool:
    """Check if COMPARISON section indicates a prior study exists."""
    if not comparison_text:
        return False
    comp = comparison_text.strip().lower()
    no_prior = re.search(
        r"(^no\b|^none\b|no.{0,15}(prior|previous|comparison)|not available)",
        comp, re.IGNORECASE,
    )
    return not no_prior and len(comp) > 2


def make_study(
    study_id: str,
    dataset: str,
    split: str,
    image_path: str,
    report_gt: str,
    findings: str,
    impression: str,
    is_followup: bool,
    prior_study: dict | None = None,
    metadata: dict | None = None,
    lateral_image_path: str = "",
) -> dict:
    """Build a standardized study dict."""
    d = {
        "study_id": study_id,
        "dataset": dataset,
        "split": split,
        "image_path": image_path,
        "lateral_image_path": lateral_image_path,
        "report_gt": clean_report(report_gt),
        "findings": clean_report(findings),
        "impression": clean_report(impression),
        "is_followup": is_followup,
        "prior_study": prior_study,
        "metadata": metadata or {},
    }
    return d


# ──────────────────────────── MIMIC-CXR ────────────────────────────

def prepare_mimic_cxr() -> list[dict]:
    """MIMIC-CXR test set: frontal views with IMPRESSION section.

    Uses enriched data from results/eval_enriched/ for prior study linking
    (image paths + reports for prior CXR studies of the same patient).
    Falls back to raw MIMIC-CXR files if enriched data is unavailable.
    """
    print("Processing MIMIC-CXR...")

    # ── Load enriched data for prior study linking ──
    enriched_by_sid = {}
    enriched_path = os.path.join(MIMIC_ENRICHED, "test_studies_enriched.json")
    if os.path.exists(enriched_path):
        with open(enriched_path) as f:
            enriched_data = json.load(f)
        for entry in enriched_data:
            enriched_by_sid[str(entry["study_id"])] = entry
        print(f"  Loaded {len(enriched_by_sid)} enriched test studies")
    else:
        print(f"  WARNING: enriched data not found at {enriched_path}, no prior linking")

    # ── Load split assignments ──
    split_map = {}  # study_id -> subject_id
    with open(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-split.csv")) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                split_map[row["study_id"]] = row["subject_id"]

    # ── Load metadata for frontal views (PA preferred over AP) + lateral ──
    study_images = {}  # study_id -> {view, dicom_id, path}
    study_laterals = {}  # study_id -> path to lateral image
    with open(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-metadata.csv")) as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            view = row.get("ViewPosition", "")
            if sid not in split_map:
                continue
            subj = split_map[sid]
            p_prefix = f"p{subj[:2]}"
            img_path = os.path.join(
                MIMIC_CXR_DIR, "files", p_prefix, f"p{subj}", f"s{sid}",
                f"{row['dicom_id']}.jpg",
            )
            if view in ("PA", "AP"):
                # Prefer PA over AP
                if sid in study_images and study_images[sid]["view"] == "PA":
                    continue
                study_images[sid] = {"view": view, "path": img_path, "dicom_id": row["dicom_id"]}
            elif view in ("LATERAL", "LL"):
                # Store first lateral per study
                if sid not in study_laterals:
                    study_laterals[sid] = img_path

    # ── Load CheXpert labels ──
    chexpert_labels = {}
    label_cols = None
    with open(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-chexpert.csv")) as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            if sid in split_map:
                if label_cols is None:
                    label_cols = [c for c in row.keys() if c not in ("subject_id", "study_id")]
                chexpert_labels[sid] = {c: row[c] for c in label_cols}

    # ── Build studies with reports ──
    studies = []
    for sid, subj in split_map.items():
        if sid not in study_images:
            continue

        # Read report
        p_prefix = f"p{subj[:2]}"
        report_path = os.path.join(MIMIC_CXR_DIR, "files", p_prefix, f"p{subj}", f"s{sid}.txt")
        if not os.path.exists(report_path):
            # Fallback to reports/ dir
            report_path = os.path.join(MIMIC_CXR_DIR, "reports", "files", p_prefix, f"p{subj}", f"s{sid}.txt")
            if not os.path.exists(report_path):
                continue

        with open(report_path) as f:
            report = f.read()

        # Must have FINDINGS (required); IMPRESSION optional
        report_upper = report.upper()
        if "FINDINGS" not in report_upper:
            continue

        findings, impression = extract_sections(report)

        # Require non-empty findings section
        if not findings.strip():
            continue

        # Determine baseline vs follow-up from COMPARISON section
        comp_match = re.search(
            r"COMPARISON[:\s]*(.*?)(?=\n\s*[A-Z]{3,}|\n\n)",
            report, re.IGNORECASE | re.DOTALL,
        )
        comparison_text = comp_match.group(1).strip() if comp_match else ""
        is_followup = has_prior_reference_comparison(comparison_text)

        # Check inline references if no COMPARISON section
        if not comp_match:
            inline = re.search(
                r"(compared? (to|with)|in comparison|prior (study|exam)|previous (study|exam))",
                report, re.IGNORECASE,
            )
            is_followup = bool(inline)

        # ── Link prior study from enriched data ──
        prior_study = None
        enriched = enriched_by_sid.get(sid)
        if enriched and enriched.get("prior_studies"):
            # Use the most recent prior (first in list, sorted by date desc)
            prior = enriched["prior_studies"][0]
            prior_img = prior.get("image_path", "")
            prior_report = prior.get("report", "")
            if prior_img and prior_report:
                prior_findings, prior_impression = extract_sections(prior_report)
                prior_study = {
                    "image_path": prior_img,
                    "report": clean_report(prior_report),
                    "findings": clean_report(prior_findings),
                    "impression": clean_report(prior_impression),
                    "study_date": prior.get("study_date", ""),
                    "study_id": str(prior.get("study_id", "")),
                }
            # If enriched data says there are priors, mark as follow-up
            if not is_followup and prior_study:
                is_followup = True

        # ── Build metadata (include admission_info if available) ──
        meta = {
            "subject_id": subj,
            "view_position": study_images[sid]["view"],
            "comparison": comparison_text,
            "chexpert_labels": chexpert_labels.get(sid, {}),
        }
        if enriched:
            meta["study_date"] = enriched.get("study_date", "")
            if enriched.get("admission_info"):
                meta["admission_info"] = enriched["admission_info"]

        studies.append(make_study(
            study_id=f"mimic_{sid}",
            dataset="mimic_cxr",
            split="test",
            image_path=study_images[sid]["path"],
            report_gt=report,
            findings=findings,
            impression=impression,
            is_followup=is_followup,
            prior_study=prior_study,
            metadata=meta,
            lateral_image_path=study_laterals.get(sid, ""),
        ))

    with_prior = sum(1 for s in studies if s["prior_study"])
    with_lateral = sum(1 for s in studies if s["lateral_image_path"])
    print(f"  MIMIC-CXR: {len(studies)} studies "
          f"({sum(1 for s in studies if not s['is_followup'])} baseline, "
          f"{sum(1 for s in studies if s['is_followup'])} follow-up, "
          f"{with_prior} with prior image+report, "
          f"{with_lateral} with lateral view)")
    return studies


# ──────────────────────────── CheXpert-Plus ────────────────────────────

def prepare_chexpert_plus() -> list[dict]:
    """CheXpert-Plus validation set with prior study linking."""
    print("Processing CheXpert-Plus (valid)...")

    csv_path = os.path.join(CHEXPERT_DIR, "df_chexpert_plus_240401.csv")

    # Index all studies by patient for prior linking
    patient_studies = defaultdict(dict)  # pid -> {order: row_data}
    valid_entries = []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row["deid_patient_id"]
            order = int(row["patient_report_date_order"])
            entry = {
                "path": row["path_to_image"],
                "report": row.get("report", ""),
                "findings": row.get("section_findings", ""),
                "impression": row.get("section_impression", ""),
                "comparison": row.get("section_comparison", ""),
                "history": row.get("section_clinical_history", ""),
                "frontal_lateral": row.get("frontal_lateral", ""),
                "ap_pa": row.get("ap_pa", ""),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "split": row.get("split", ""),
                "pid": pid,
                "order": order,
            }
            patient_studies[pid][order] = entry
            if row.get("split") == "valid":
                valid_entries.append(entry)

    studies = []
    seen_study_ids = set()
    for entry in valid_entries:
        # Filter: frontal only
        if entry["frontal_lateral"] and "frontal" not in entry["frontal_lateral"].lower():
            continue

        # Must have findings (required); impression optional
        findings_check = entry["findings"].strip()
        if not findings_check:
            continue
        impression = entry["impression"].strip()

        # Deduplicate: one image per study (CheXpert may have multiple frontal views)
        study_key = f"{Path(entry['path']).parent.parent.name}_{Path(entry['path']).parent.name}"
        if study_key in seen_study_ids:
            continue
        seen_study_ids.add(study_key)

        findings = entry["findings"].strip()
        comparison = entry["comparison"].strip()
        is_followup = has_prior_reference_comparison(comparison)

        # Build image path
        image_path = os.path.join(CHEXPERT_DIR, entry["path"])

        # CheXpert-Plus valid set is too small (202 studies) for a meaningful
        # follow-up track — treat all as baseline
        prior_study = None
        is_followup = False

        report_gt = ""
        if findings:
            report_gt = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            report_gt = f"IMPRESSION:\n{impression}"

        studies.append(make_study(
            study_id=f"chexpert_{Path(entry['path']).parent.parent.name}_{Path(entry['path']).parent.name}",
            dataset="chexpert_plus",
            split="valid",
            image_path=image_path,
            report_gt=report_gt,
            findings=findings,
            impression=impression,
            is_followup=False,
            prior_study=None,
            metadata={
                "patient_id": entry["pid"],
                "report_date_order": entry["order"],
                "view_position": entry["ap_pa"],
                "comparison": comparison,
                "age": entry["age"],
                "sex": entry["sex"],
            },
        ))

    print(f"  CheXpert-Plus: {len(studies)} studies "
          f"({sum(1 for s in studies if not s['is_followup'])} baseline, "
          f"{sum(1 for s in studies if s['is_followup'])} follow-up, "
          f"{sum(1 for s in studies if s['prior_study'])} with prior image+report)")
    return studies


# ──────────────────────────── ReXGradient ────────────────────────────

def prepare_rexgradient() -> list[dict]:
    """ReXGradient test set with cross-split prior linking."""
    print("Processing ReXGradient...")

    # Load all metadata across splits for prior linking
    # Use *_view_position.json which has proper view labels (from HF repo)
    all_studies_by_patient = defaultdict(list)
    vp_snapshot = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--rajpurkarlab--ReXGradient-160K", "snapshots",
        "b9277dd929190931f5a8b6ae19f3e3c8aae2f1f7", "metadata",
    )
    for split_name in ("train", "valid", "test"):
        csv_path = os.path.join(REXGRADIENT_META, f"{split_name}_metadata.csv")
        # Prefer view_position JSON (has proper labels), fall back to regular
        json_path = os.path.join(vp_snapshot, f"{split_name}_metadata_view_position.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(REXGRADIENT_META, f"{split_name}_metadata.json")

        # CSV has reports, JSON has image paths + view positions
        csv_rows = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                csv_rows[row["id"]] = row

        with open(json_path) as f:
            json_data = json.load(f)

        for study_id, meta in json_data.items():
            csv_row = csv_rows.get(study_id, {})
            pid = study_id.split("_")[0]
            image_paths = meta.get("ImagePath", [])
            view_positions = meta.get("ImageViewPosition", [])

            # Find frontal image (PA/AP only) + lateral, skip lateral-only studies
            frontal_path = None
            frontal_view = None
            lateral_path = None
            for img_p, vp in zip(image_paths, view_positions):
                if vp in ("PA", "AP", "POSTERO_ANTERIOR", "AP (KUB)"):
                    if not frontal_path:
                        rel = img_p.replace("../deid_png/", "")
                        frontal_path = os.path.join(REXGRADIENT_IMAGES, rel)
                        frontal_view = vp
                elif vp in ("LATERAL", "LL"):
                    if not lateral_path:
                        rel = img_p.replace("../deid_png/", "")
                        lateral_path = os.path.join(REXGRADIENT_IMAGES, rel)

            if not frontal_path:
                continue  # no frontal image available, skip

            all_studies_by_patient[pid].append({
                "study_id": study_id,
                "split": split_name,
                "date": str(csv_row.get("StudyDate", meta.get("StudyDate", ""))),
                "image_path": frontal_path,
                "lateral_image_path": lateral_path or "",
                "view": frontal_view,
                "findings": csv_row.get("Findings", ""),
                "impression": csv_row.get("Impression", ""),
                "comparison": csv_row.get("Comparison", ""),
                "indication": csv_row.get("Indication", ""),
                "age": csv_row.get("PatientAge", ""),
                "sex": csv_row.get("PatientSex", ""),
            })

    # Sort each patient's studies by date
    for pid in all_studies_by_patient:
        all_studies_by_patient[pid].sort(key=lambda x: x["date"])

    # Build test studies
    studies = []
    for pid, patient_list in all_studies_by_patient.items():
        for i, entry in enumerate(patient_list):
            if entry["split"] != "test":
                continue

            findings = entry["findings"].strip()
            impression = entry["impression"].strip()

            # Must have Findings (required); Impression optional
            if not findings:
                continue

            if not entry["image_path"]:
                continue

            comparison = entry["comparison"].strip()
            is_followup = has_prior_reference_comparison(comparison)

            # Find prior study (closest earlier date, any split)
            prior_study = None
            for j in range(i - 1, -1, -1):
                prior = patient_list[j]
                if prior["date"] < entry["date"] and prior["image_path"]:
                    prior_findings = prior["findings"].strip()
                    prior_impression = prior["impression"].strip()
                    prior_report = ""
                    if prior_findings and prior_impression:
                        prior_report = f"FINDINGS:\n{prior_findings}\n\nIMPRESSION:\n{prior_impression}"
                    elif prior_impression:
                        prior_report = f"IMPRESSION:\n{prior_impression}"
                    if prior_report:
                        prior_study = {
                            "image_path": prior["image_path"],
                            "report": clean_report(prior_report),
                            "findings": clean_report(prior_findings),
                            "impression": clean_report(prior_impression),
                            "study_date": prior["date"],
                        }
                    break

            report_gt = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

            studies.append(make_study(
                study_id=f"rexgrad_{entry['study_id']}",
                dataset="rexgradient",
                split="test",
                image_path=entry["image_path"],
                report_gt=report_gt,
                findings=findings,
                impression=impression,
                is_followup=is_followup,
                prior_study=prior_study,
                lateral_image_path=entry.get("lateral_image_path", ""),
                metadata={
                    "patient_id": pid,
                    "view_position": entry["view"],
                    "study_date": entry["date"],
                    "comparison": comparison,
                    "indication": entry["indication"],
                    "age": entry["age"],
                    "sex": entry["sex"],
                },
            ))

    with_lateral = sum(1 for s in studies if s["lateral_image_path"])
    print(f"  ReXGradient: {len(studies)} studies "
          f"({sum(1 for s in studies if not s['is_followup'])} baseline, "
          f"{sum(1 for s in studies if s['is_followup'])} follow-up, "
          f"{sum(1 for s in studies if s['prior_study'])} with prior image+report, "
          f"{with_lateral} with lateral view)")
    return studies


# ──────────────────────────── IU-Xray ────────────────────────────

def prepare_iu_xray() -> list[dict]:
    """IU-Xray test set (590 studies defined by ReXrank)."""
    print("Processing IU-Xray...")

    # Load ReXrank test IDs
    with open(os.path.join(REXRANK_DIR, "data", "submission_example.json")) as f:
        rexrank_data = json.load(f)

    # Load projections to map case_id -> image files
    projections = defaultdict(list)  # uid -> [(filename, projection)]
    with open(os.path.join(IU_XRAY_DIR, "indiana_projections.csv")) as f:
        for row in csv.DictReader(f):
            projections[row["uid"]].append((row["filename"], row["projection"]))

    # Load reports
    reports = {}
    with open(os.path.join(IU_XRAY_DIR, "indiana_reports.csv")) as f:
        for row in csv.DictReader(f):
            reports[row["uid"]] = row

    # Build filename index: base_name -> [(filename, projection)]
    img_dir = os.path.join(IU_XRAY_DIR, "images", "images_normalized")
    all_img_files = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()

    # Build frontal + lateral image lookup from projections CSV (reliable view labels)
    frontal_by_uid = {}  # uid -> first frontal filename
    lateral_by_uid = {}  # uid -> first lateral filename
    for uid, proj_list in projections.items():
        for fn, proj in proj_list:
            if proj == "Frontal" and fn in all_img_files and uid not in frontal_by_uid:
                frontal_by_uid[uid] = fn
            elif proj == "Lateral" and fn in all_img_files and uid not in lateral_by_uid:
                lateral_by_uid[uid] = fn

    studies = []
    for case_id, entry in rexrank_data.items():
        # Get report from ReXrank data (already has findings/impression)
        findings = entry.get("section_findings", "").strip()
        impression = entry.get("section_impression", "").strip()

        # Must have findings (required); impression optional
        if not findings:
            continue

        # Map CXR{uid}_IM-{id} to uid
        uid_match = re.match(r"CXR(\d+)_", case_id)
        uid = uid_match.group(1) if uid_match else None

        # Use projections CSV to find frontal image (not filename suffix heuristic)
        image_path = None
        if uid and uid in frontal_by_uid:
            image_path = os.path.join(img_dir, frontal_by_uid[uid])
        else:
            # Fallback: match by case_id prefix, pick first file in projections
            # labeled as Frontal
            base = case_id.replace("CXR", "")
            matching_files = sorted(f for f in all_img_files if f.startswith(base))
            for fn in matching_files:
                # Check projections CSV for this filename
                for p_uid, proj_list in projections.items():
                    for p_fn, proj in proj_list:
                        if p_fn == fn and proj == "Frontal":
                            image_path = os.path.join(img_dir, fn)
                            break
                    if image_path:
                        break
                if image_path:
                    break
            # Last resort: skip study if no frontal view found
            if not image_path:
                continue

        report_gt = ""
        if findings:
            report_gt = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        else:
            report_gt = f"IMPRESSION:\n{impression}"

        # IU-Xray has no patient linking, so all baseline
        comparison = ""
        if uid and uid in reports:
            comparison = reports[uid].get("comparison", "").strip()

        # Lateral view
        lateral_path = ""
        if uid and uid in lateral_by_uid:
            lateral_path = os.path.join(img_dir, lateral_by_uid[uid])

        studies.append(make_study(
            study_id=f"iu_{case_id}",
            dataset="iu_xray",
            split="test",
            image_path=image_path,
            report_gt=report_gt,
            findings=findings,
            impression=impression,
            is_followup=False,  # No prior linking possible
            lateral_image_path=lateral_path,
            metadata={
                "case_id": case_id,
                "comparison": comparison,
                "indication": entry.get("context", ""),
            },
        ))

    with_lateral = sum(1 for s in studies if s["lateral_image_path"])
    print(f"  IU-Xray: {len(studies)} studies (all baseline, no prior linking, "
          f"{with_lateral} with lateral view)")
    return studies


# ──────────────────────────── PadChest-GR ────────────────────────────

def prepare_padchest_gr() -> list[dict]:
    """PadChest-GR test set (20% split from paper)."""
    print("Processing PadChest-GR...")

    # Load grounded reports
    with open(os.path.join(PADCHEST_GR_DIR, "grounded_reports_20240819.json")) as f:
        gr_data = json.load(f)

    # Load master table for split info
    study_meta = {}
    with open(os.path.join(PADCHEST_GR_DIR, "master_table.csv")) as f:
        for row in csv.DictReader(f):
            sid = row["StudyID"]
            if sid not in study_meta:
                study_meta[sid] = {
                    "split": row["split"],
                    "patient_id": row.get("PatientID", ""),
                    "age": row.get("PatientAge", ""),
                    "sex": row.get("PatientSex_DICOM", ""),
                    "study_date": row.get("StudyDate_DICOM", ""),
                }

    # Build image path lookup
    # Images extracted to Padchest_GR_files/PadChest_GR/ subfolder
    img_base = os.path.join(PADCHEST_GR_DIR, "Padchest_GR_files", "PadChest_GR")

    # Index grounded reports by StudyID
    gr_by_study = {}
    for entry in gr_data:
        sid = entry["StudyID"]
        if sid not in gr_by_study:
            gr_by_study[sid] = entry


    studies = []
    for sid, entry in gr_by_study.items():
        meta = study_meta.get(sid, {})
        if meta.get("split") != "test":
            continue

        image_id = entry["ImageID"]

        # Try multiple image locations
        image_path = None
        for base_dir in [img_base, PADCHEST_GR_DIR]:
            candidate = os.path.join(base_dir, image_id)
            if os.path.exists(candidate):
                image_path = candidate
                break
        # Also search recursively if not found
        if not image_path:
            for base_dir in [img_base]:
                for root, dirs, files in os.walk(base_dir):
                    if image_id in files:
                        image_path = os.path.join(root, image_id)
                        break
                if image_path:
                    break

        if not image_path:
            # Image not yet extracted, store expected path
            image_path = os.path.join(img_base, image_id)

        # Build report from grounded findings
        positive_findings = []
        negative_findings = []
        groundings = []

        for finding in entry.get("findings", []):
            sentence = finding.get("sentence_en", "")
            if not sentence:
                continue
            if finding.get("abnormal"):
                positive_findings.append(sentence)
                boxes = finding.get("boxes", [])
                if boxes:
                    groundings.append({
                        "sentence": sentence,
                        "boxes": boxes,
                        "labels": finding.get("labels", []),
                        "locations": finding.get("locations", []),
                        "progression": finding.get("progression"),
                    })
            else:
                negative_findings.append(sentence)

        findings_text = " ".join(positive_findings + negative_findings)
        # PadChest-GR doesn't have separate impression; use findings as full report
        impression_text = ""
        report_gt = findings_text

        # PadChest-GR: treat all as baseline — GT reports are finding-level
        # sentences without comparison language, and original PadChest prior
        # reports are stemmed Spanish (not usable)
        prior_study = None
        is_followup = False

        studies.append(make_study(
            study_id=f"padchest_{sid}",
            dataset="padchest_gr",
            split="test",
            image_path=image_path,
            report_gt=report_gt,
            findings=findings_text,
            impression=impression_text,
            is_followup=is_followup,
            prior_study=prior_study,
            metadata={
                "patient_id": meta.get("patient_id", ""),
                "age": meta.get("age", ""),
                "sex": meta.get("sex", ""),
                "study_date": meta.get("study_date", ""),
                "groundings": groundings,
            },
        ))

    print(f"  PadChest-GR: {len(studies)} studies "
          f"({sum(1 for s in studies if not s['is_followup'])} baseline, "
          f"{sum(1 for s in studies if s['is_followup'])} follow-up)")
    return studies


# ──────────────────────────── main ────────────────────────────

def finalize_studies(studies: list[dict]) -> list[dict]:
    """Drop orphan follow-ups and add eval_track field.

    Orphan follow-ups have is_followup=True but no prior_study linked —
    the GT report may reference priors the agent can't see, so drop them.
    """
    out = []
    dropped = 0
    for s in studies:
        if s["is_followup"] and not s.get("prior_study"):
            dropped += 1
            continue
        if s["is_followup"]:
            s["eval_track"] = "followup"
        else:
            s["eval_track"] = "baseline"
            s["prior_study"] = None  # baseline should not carry prior
        out.append(s)
    if dropped:
        print(f"  Dropped {dropped} orphan follow-ups (no prior available)")
    return out


def save_dataset(studies: list[dict], name: str, output_dir: str):
    """Save dataset JSON and print stats."""
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(studies, f, indent=2)
    print(f"  Saved {len(studies)} studies to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation datasets")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--datasets", default="all",
        help="Comma-separated list: mimic_cxr,chexpert_plus,rexgradient,iu_xray (default=all 4). padchest_gr available but excluded by default.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # PadChest-GR excluded by default: reports are translated sentence fragments
    # with no FINDINGS/IMPRESSION structure, incompatible with report generation
    # metrics.  Can still be run explicitly via --datasets padchest_gr.
    datasets_to_run = args.datasets.split(",") if args.datasets != "all" else [
        "mimic_cxr", "chexpert_plus", "rexgradient", "iu_xray",
    ]

    all_studies = []
    summary = {}

    processors = {
        "mimic_cxr": (prepare_mimic_cxr, "mimic_cxr_test"),
        "chexpert_plus": (prepare_chexpert_plus, "chexpert_plus_valid"),
        "rexgradient": (prepare_rexgradient, "rexgradient_test"),
        "iu_xray": (prepare_iu_xray, "iu_xray_test"),
        "padchest_gr": (prepare_padchest_gr, "padchest_gr_test"),
    }

    for ds_name in datasets_to_run:
        if ds_name not in processors:
            print(f"Unknown dataset: {ds_name}")
            continue
        func, file_name = processors[ds_name]
        try:
            studies = finalize_studies(func())
            save_dataset(studies, file_name, args.output)
            all_studies.extend(studies)

            baseline = [s for s in studies if s.get("eval_track") == "baseline"]
            followup = [s for s in studies if s.get("eval_track") == "followup"]
            with_prior = [s for s in followup if s.get("prior_study")]

            with_lateral = [s for s in studies if s.get("lateral_image_path")]

            summary[ds_name] = {
                "file": f"{file_name}.json",
                "split": studies[0]["split"] if studies else "",
                "total": len(studies),
                "baseline": len(baseline),
                "followup": len(followup),
                "with_prior": len(with_prior),
                "with_lateral": len(with_lateral),
                "has_findings": sum(1 for s in studies if s["findings"]),
                "has_impression": sum(1 for s in studies if s["impression"]),
            }
        except Exception as e:
            print(f"  ERROR processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            summary[ds_name] = {"error": str(e)}

    # Save summary
    total_baseline = sum(d.get("baseline", 0) for d in summary.values() if "error" not in d)
    total_followup = sum(d.get("followup", 0) for d in summary.values() if "error" not in d)
    total_all = sum(d.get("total", 0) for d in summary.values() if "error" not in d)

    summary["_total"] = {
        "studies": total_all,
        "baseline": total_baseline,
        "followup": total_followup,
    }

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_all} studies ({total_baseline} baseline, {total_followup} follow-up)")
    print(f"Summary saved to {summary_path}")

    # Print table
    print(f"\n{'Dataset':<20} {'Split':<8} {'Total':>7} {'Base':>7} {'F/U':>7} {'Prior':>7} {'Lat':>7}")
    print("-" * 68)
    for ds_name, info in summary.items():
        if ds_name.startswith("_") or "error" in info:
            continue
        print(f"{ds_name:<20} {info['split']:<8} {info['total']:>7} "
              f"{info['baseline']:>7} {info['followup']:>7} {info.get('with_prior', 0):>7} "
              f"{info.get('with_lateral', 0):>7}")
    print("-" * 68)
    print(f"{'TOTAL':<29} {total_all:>7} {total_baseline:>7} {total_followup:>7}")


if __name__ == "__main__":
    main()
