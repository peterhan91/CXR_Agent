#!/usr/bin/env python3
"""
Prepare enriched MIMIC-CXR validation & test sets for CXR Agent evaluation.

Combines MIMIC-CXR image/report data with MIMIC-IV clinical context:
- Patient admission info (history, exam, meds, diagnoses, labs)
- Prior CXR studies (image paths + reports) for temporal comparison
- CheXpert pathology labels

Output: JSON files with enriched study entries for agent evaluation.

Usage:
    python scripts/prepare_mimic_studies.py \
        --mimic_cxr_dir /path/to/MIMIC-CXR-JPEG \
        --mimic_iv_dir /path/to/mimiciv/3.1 \
        --output results/eval_enriched/ \
        --splits validate,test \
        --skip_labs
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Report Parsing (reused from eval_mimic.py) ─────────────────────────────


def parse_report_sections(text: str) -> tuple:
    """Extract FINDINGS and IMPRESSION from raw MIMIC-CXR report text."""
    findings = ""
    impression = ""

    m = re.search(
        r"FINDINGS?:?\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|RECOMMENDATION)\s*:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        findings = m.group(1).strip()

    m = re.search(
        r"(?:IMPRESSION|CONCLUSION):?\s*(.*?)(?=(?:RECOMMENDATION|NOTIFICATION|ADDENDUM)\s*:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        impression = m.group(1).strip()

    return findings, impression


def format_gt_report(findings: str, impression: str, raw_text: str) -> str:
    """Format ground truth report. Prefers structured sections, falls back to raw."""
    parts = []
    if findings:
        parts.append(f"FINDINGS:\n{findings}")
    if impression:
        parts.append(f"IMPRESSION:\n{impression}")
    if parts:
        return "\n\n".join(parts)
    return raw_text.strip()


def _find_csv(directory: Path, stem: str) -> Path:
    """Find a CSV file by stem, trying .csv.gz then .csv."""
    for ext in [".csv.gz", ".csv"]:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    return None


# ─── Discharge Note Section Extraction (from MIMIC-Plain patterns) ──────────


def _regex_extracter(text, regex):
    """Extract text using regex. Returns (matched_text, True) or (original, False)."""
    try:
        return re.search(regex, text).group(0), True
    except Exception:
        return text, False


def _last_substring_index(larger_string, substring):
    """Find the last occurrence of substring in larger_string."""
    last_index = -1
    while True:
        index = larger_string.find(substring, last_index + 1)
        if index == -1:
            break
        last_index = index
    return last_index


def _extract_section(text: str, start_headers: list, end_headers: list) -> str:
    """Generic helper to extract a section between given headers."""
    if not text:
        return ""
    tl = text.lower()

    start_idx = -1
    start_len = 0
    for sh in start_headers:
        pos = _last_substring_index(tl, sh.lower())
        if pos != -1 and pos >= start_idx:
            start_idx = pos
            start_len = len(sh)
    if start_idx == -1:
        return ""

    end_idx = len(text)
    for eh in end_headers:
        pos = tl.find(eh.lower(), start_idx + start_len)
        if pos != -1:
            end_idx = min(end_idx, pos)

    content = text[start_idx + start_len : end_idx].strip()
    return content


def extract_history(text: str) -> str:
    """Extract History of Present Illness from discharge summary."""
    text = text.replace("\n", " ")
    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "(?:pertinent|___) results:",
        "hospital course:",
    ]
    while not success and i < len(pe_strings):
        regex = re.compile(
            f"(?:history|___) of present(?:ing)? illness:.*?{pe_strings[i]}",
            re.IGNORECASE | re.DOTALL,
        )
        text, success = _regex_extracter(text, regex)
        i += 1
    if not success:
        return ""

    text = re.sub(
        re.compile("history of present(?:ing)? illness:", re.IGNORECASE), "", text
    )
    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)
    return text.strip()


def extract_physical_examination(text: str) -> str:
    """Extract Physical Examination from discharge summary."""
    text = text.replace("\n", " ")
    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "pertinent results:",
    ]
    while not success and i < len(pe_strings):
        terminal_str = "pertinent results:"
        if terminal_str not in text.lower():
            terminal_str = "brief hospital course:"
        regex = re.compile(
            f"{pe_strings[i]}.*?{terminal_str}", re.IGNORECASE | re.DOTALL
        )
        text, success = _regex_extracter(text, regex)
        i += 1
    if not success:
        return ""

    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)
    text = re.sub(re.compile("pertinent results:", re.IGNORECASE), "", text)
    text = re.sub(re.compile("brief hospital course:", re.IGNORECASE), "", text)

    text = re.sub(re.compile("at discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("upon discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("on discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("discharge.*", re.IGNORECASE), "", text)
    return text.strip()


def extract_chief_complaint(text: str) -> str:
    """Extract Chief Complaint from discharge summary."""
    regex = re.compile(
        "(?:chief|___) complaint:(.*)major (?:surgical|___)",
        re.IGNORECASE | re.DOTALL,
    )
    match = regex.findall(text)
    if match:
        return match[0].strip()
    return ""


def extract_medications_on_admission(text: str) -> str:
    """Extract Medications on Admission from discharge summary."""
    start_headers = [
        "Medications on Admission:",
        "Home Medications:",
        "Admission Medications:",
        "Meds on Admission:",
        "Prior to Admission Medications:",
    ]
    end_headers = [
        "Discharge Medications:",
        "Medications at Discharge:",
        "Discharge Medication:",
        "Discharge Diagnosis:",
        "Discharge Condition:",
        "Discharge Disposition:",
        "Discharge Instructions:",
        "Followup Instructions:",
        "Brief Hospital Course:",
        "Hospital Course:",
        "Pertinent Results:",
    ]
    return _extract_section(text, start_headers, end_headers)


def extract_discharge_diagnosis(text: str) -> str:
    """Extract Discharge Diagnosis from discharge summary."""
    start_headers = ["discharge diagnosis:", "___ diagnosis:"]
    end_headers = [
        "discharge condition:",
        "___ condition:",
        "condition:",
        "procedure:",
        "procedures:",
        "invasive procedure on this admission:",
    ]
    tl = text.lower()

    start = 0
    for sh in start_headers:
        pos = _last_substring_index(tl, sh)
        if pos != -1:
            start = max(start, pos + len(sh))
    if not start:
        # Last resort: empty string header
        sh = "\n___:"
        pos = _last_substring_index(tl, sh)
        if pos != -1:
            start = pos
        else:
            return ""

    end = 0
    for eh in end_headers:
        pos = _last_substring_index(tl, eh)
        if pos != -1:
            end = max(end, pos)
        if end:
            break
    if not end:
        return ""

    return text[start:end].strip()


# ─── StudyDate Conversion ───────────────────────────────────────────────────


def convert_study_date(date_int) -> str:
    """Convert StudyDate integer 21800506 -> string '2180-05-06'."""
    s = str(int(date_int))
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


# ─── Phase 1: Load MIMIC-CXR Metadata ──────────────────────────────────────


def load_cxr_metadata(mimic_cxr_dir: Path, splits: list[str]):
    """Load split, metadata, and chexpert CSVs. Filter to target splits.

    Returns:
        studies_df: DataFrame with one row per study (best frontal image),
                    columns: dicom_id, subject_id, study_id, split, ViewPosition, StudyDate
        chexpert_df: DataFrame with CheXpert labels per study
    """
    logger.info("Phase 1: Loading MIMIC-CXR metadata...")

    # Load split CSV
    split_path = _find_csv(mimic_cxr_dir, "mimic-cxr-2.0.0-split")
    if not split_path:
        logger.error(f"Split CSV not found in {mimic_cxr_dir}")
        sys.exit(1)
    split_df = pd.read_csv(split_path)
    logger.info(f"  Split CSV: {len(split_df)} rows")

    # Load metadata CSV
    meta_path = _find_csv(mimic_cxr_dir, "mimic-cxr-2.0.0-metadata")
    if not meta_path:
        logger.error(f"Metadata CSV not found in {mimic_cxr_dir}")
        sys.exit(1)
    meta_df = pd.read_csv(meta_path)
    logger.info(f"  Metadata CSV: {len(meta_df)} rows")

    # Load CheXpert CSV
    chexpert_path = _find_csv(mimic_cxr_dir, "mimic-cxr-2.0.0-chexpert")
    chexpert_df = None
    if chexpert_path:
        chexpert_df = pd.read_csv(chexpert_path)
        logger.info(f"  CheXpert CSV: {len(chexpert_df)} rows")

    # Filter to target splits
    target_df = split_df[split_df["split"].isin(splits)].copy()
    logger.info(f"  Target splits {splits}: {len(target_df)} images")

    # Merge with metadata for ViewPosition and StudyDate
    target_meta = target_df.merge(
        meta_df[["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate"]],
        on=["dicom_id", "subject_id", "study_id"],
        how="left",
    )

    # Filter for frontal views only
    frontal = target_meta[target_meta["ViewPosition"].isin(["PA", "AP"])].copy()
    logger.info(f"  Frontal images: {len(frontal)}")

    # Pick best frontal per study (PA > AP)
    frontal["_priority"] = frontal["ViewPosition"].map({"PA": 0, "AP": 1})
    studies_df = frontal.sort_values("_priority").drop_duplicates("study_id", keep="first")
    studies_df = studies_df.drop(columns=["_priority"])
    logger.info(f"  Unique studies with frontal view: {len(studies_df)}")
    logger.info(f"  Unique subjects: {studies_df['subject_id'].nunique()}")

    return studies_df, chexpert_df, split_df, meta_df


# ─── Phase 2: Load CXR Reports ─────────────────────────────────────────────


def load_reports(mimic_cxr_dir: Path, studies_df: pd.DataFrame) -> dict:
    """Read report .txt files for all target studies.

    Returns:
        reports: dict mapping study_id -> {"raw": str, "findings": str, "impression": str, "formatted": str}
    """
    logger.info("Phase 2: Loading CXR reports...")
    reports_base = mimic_cxr_dir / "files"
    reports = {}
    missing = 0

    for _, row in studies_df.iterrows():
        sid = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        prefix = f"p{sid[:2]}"
        rpt_path = reports_base / prefix / f"p{sid}" / f"s{stid}.txt"

        if not rpt_path.exists():
            missing += 1
            continue

        raw = rpt_path.read_text()
        findings, impression = parse_report_sections(raw)
        formatted = format_gt_report(findings, impression, raw)
        reports[int(row["study_id"])] = {
            "raw": raw,
            "findings": findings,
            "impression": impression,
            "formatted": formatted,
        }

    logger.info(f"  Loaded {len(reports)} reports, {missing} missing")
    return reports


# ─── Phase 3: Build Prior Study Index ──────────────────────────────────────


def build_prior_index(
    mimic_cxr_dir: Path,
    target_subjects: set,
    split_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> dict:
    """Build prior study index for target subjects across ALL splits.

    Returns:
        prior_index: dict mapping subject_id -> list of {study_id, study_date, dicom_id, image_path, report}
                     sorted by study_date descending
    """
    logger.info("Phase 3: Building prior study index...")

    # Get ALL studies for target subjects (including train)
    all_for_subjects = split_df[split_df["subject_id"].isin(target_subjects)].copy()
    logger.info(f"  All images for target subjects: {len(all_for_subjects)}")

    # Merge with metadata
    all_meta = all_for_subjects.merge(
        meta_df[["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate"]],
        on=["dicom_id", "subject_id", "study_id"],
        how="left",
    )

    # Filter frontal views, pick best per study
    frontal = all_meta[all_meta["ViewPosition"].isin(["PA", "AP"])].copy()
    frontal["_priority"] = frontal["ViewPosition"].map({"PA": 0, "AP": 1})
    all_studies = frontal.sort_values("_priority").drop_duplicates("study_id", keep="first")
    all_studies = all_studies.drop(columns=["_priority"])
    logger.info(f"  All frontal studies for target subjects: {len(all_studies)}")

    reports_base = mimic_cxr_dir / "files"
    prior_index = {}

    for _, row in all_studies.iterrows():
        subj = int(row["subject_id"])
        stid = int(row["study_id"])
        did = str(row["dicom_id"])
        sid_str = str(subj)
        stid_str = str(stid)
        prefix = f"p{sid_str[:2]}"

        # Load report
        rpt_path = reports_base / prefix / f"p{sid_str}" / f"s{stid_str}.txt"
        report = ""
        if rpt_path.exists():
            raw = rpt_path.read_text()
            findings, impression = parse_report_sections(raw)
            report = format_gt_report(findings, impression, raw)

        # Construct image path
        img_path = str(
            mimic_cxr_dir / "files" / prefix / f"p{sid_str}" / f"s{stid_str}" / f"{did}.jpg"
        )

        study_date = convert_study_date(row["StudyDate"]) if pd.notna(row["StudyDate"]) else ""

        entry = {
            "study_id": stid_str,
            "study_date": study_date,
            "dicom_id": did,
            "image_path": img_path,
            "report": report,
        }

        if subj not in prior_index:
            prior_index[subj] = []
        prior_index[subj].append(entry)

    # Sort each subject's studies by date descending
    for subj in prior_index:
        prior_index[subj].sort(key=lambda x: x["study_date"], reverse=True)

    logger.info(f"  Built prior index for {len(prior_index)} subjects")
    total_studies = sum(len(v) for v in prior_index.values())
    logger.info(f"  Total studies in prior index: {total_studies}")

    return prior_index


def get_prior_studies(
    prior_index: dict, subject_id: int, study_id: str, study_date: str
) -> list:
    """Get all prior studies for a subject that are earlier than the target study."""
    all_studies = prior_index.get(subject_id, [])
    priors = []
    for s in all_studies:
        # Exclude the current study and any studies on or after the target date
        if s["study_id"] == study_id:
            continue
        if s["study_date"] and study_date and s["study_date"] < study_date:
            priors.append(s)
    return priors


# ─── Phase 4: Link to MIMIC-IV Admissions ──────────────────────────────────


def link_admissions(
    mimic_iv_dir: Path, studies_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load admissions and patients, match studies to admissions.

    Match via subject_id + StudyDate in [admittime - 1day, dischtime + 1day].

    Returns:
        admissions_df: filtered admissions for target subjects
        patients_df: filtered patients for target subjects
    """
    logger.info("Phase 4: Linking to MIMIC-IV admissions...")

    target_subjects = set(studies_df["subject_id"].unique())

    # Load admissions
    adm_path = mimic_iv_dir / "hosp" / "admissions.csv"
    admissions_df = pd.read_csv(adm_path)
    admissions_df = admissions_df[admissions_df["subject_id"].isin(target_subjects)].copy()
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
    admissions_df["dischtime"] = pd.to_datetime(admissions_df["dischtime"])
    logger.info(f"  Admissions for target subjects: {len(admissions_df)}")

    # Load patients
    pat_path = mimic_iv_dir / "hosp" / "patients.csv"
    patients_df = pd.read_csv(pat_path)
    patients_df = patients_df[patients_df["subject_id"].isin(target_subjects)].copy()
    logger.info(f"  Patients for target subjects: {len(patients_df)}")

    return admissions_df, patients_df


def match_study_to_admission(
    subject_id: int, study_date_str: str, admissions_df: pd.DataFrame
) -> dict | None:
    """Match a study to an admission via date overlap.

    Returns admission row as dict, or None if no match.
    """
    if not study_date_str:
        return None

    study_dt = pd.Timestamp(study_date_str)
    subj_adm = admissions_df[admissions_df["subject_id"] == subject_id]

    for _, adm in subj_adm.iterrows():
        admit = adm["admittime"]
        disch = adm["dischtime"]
        # Allow 1-day buffer on each side
        if pd.notna(admit) and pd.notna(disch):
            if (admit - pd.Timedelta(days=1)) <= study_dt <= (disch + pd.Timedelta(days=1)):
                return adm.to_dict()
        elif pd.notna(admit):
            # No discharge time yet — match if study is within 30 days of admit
            if (admit - pd.Timedelta(days=1)) <= study_dt <= (admit + pd.Timedelta(days=30)):
                return adm.to_dict()

    return None


# ─── Phase 5: Extract Admission Info ───────────────────────────────────────


def load_discharge_notes(mimic_iv_dir: Path, target_hadm_ids: set) -> dict:
    """Load discharge notes for target admissions (chunked read for 3.5GB file).

    Returns:
        notes: dict mapping hadm_id -> discharge note text
    """
    logger.info("Phase 5a: Loading discharge notes (chunked)...")

    # Try uncompressed first, then compressed
    note_path = mimic_iv_dir / "note" / "discharge.csv"
    if not note_path.exists():
        note_path = mimic_iv_dir / "note" / "discharge.csv.gz"
    if not note_path.exists():
        logger.warning(f"  Discharge notes not found at {mimic_iv_dir / 'note'}")
        return {}

    notes = {}
    chunk_size = 10000
    chunks_read = 0

    for chunk in pd.read_csv(note_path, chunksize=chunk_size):
        chunks_read += 1
        matched = chunk[chunk["hadm_id"].isin(target_hadm_ids)]
        for _, row in matched.iterrows():
            hadm_id = int(row["hadm_id"])
            if hadm_id not in notes:  # Keep first note per hadm_id
                notes[hadm_id] = str(row["text"])
        if chunks_read % 100 == 0:
            logger.info(f"    Read {chunks_read} chunks, found {len(notes)} notes so far...")

    logger.info(f"  Loaded {len(notes)} discharge notes from {chunks_read} chunks")
    return notes


def extract_admission_sections(note_text: str) -> dict:
    """Extract clinical sections from a discharge note.

    Returns dict with: patient_history, physical_examination, chief_complaint,
                       medications_on_admission, discharge_diagnosis
    """
    return {
        "patient_history": extract_history(note_text),
        "physical_examination": extract_physical_examination(note_text),
        "chief_complaint": extract_chief_complaint(note_text),
        "medications_on_admission": extract_medications_on_admission(note_text),
        "discharge_diagnosis": extract_discharge_diagnosis(note_text),
    }


def load_icd_diagnoses(mimic_iv_dir: Path, target_hadm_ids: set) -> dict:
    """Load ICD diagnoses for target admissions.

    Returns:
        icd_by_hadm: dict mapping hadm_id -> list of {code, version, description}
    """
    logger.info("Phase 5b: Loading ICD diagnoses...")

    diag_path = mimic_iv_dir / "hosp" / "diagnoses_icd.csv"
    diag_df = pd.read_csv(diag_path)
    diag_df = diag_df[diag_df["hadm_id"].isin(target_hadm_ids)]

    # Load descriptions
    desc_path = mimic_iv_dir / "hosp" / "d_icd_diagnoses.csv"
    desc_df = pd.read_csv(desc_path)

    # Merge
    merged = diag_df.merge(desc_df, on=["icd_code", "icd_version"], how="left")

    icd_by_hadm = {}
    for hadm_id, group in merged.groupby("hadm_id"):
        diagnoses = []
        for _, row in group.sort_values("seq_num").iterrows():
            diagnoses.append({
                "code": str(row["icd_code"]),
                "version": int(row["icd_version"]),
                "description": str(row["long_title"]) if pd.notna(row["long_title"]) else "",
            })
        icd_by_hadm[int(hadm_id)] = diagnoses

    logger.info(f"  Loaded ICD diagnoses for {len(icd_by_hadm)} admissions")
    return icd_by_hadm


# Key lab item IDs relevant to CXR interpretation
CXR_LAB_ITEMIDS = {
    51006: "BUN",
    50912: "Creatinine",
    51265: "Platelet Count",
    51301: "WBC",
    51222: "Hemoglobin",
    51249: "MCHC",
    50971: "Potassium",
    50983: "Sodium",
    50902: "Chloride",
    50882: "Bicarbonate",
    50868: "Anion Gap",
    50931: "Glucose",
    51003: "Troponin T",
    50963: "BNP",
    50889: "CRP",
    50813: "Lactate",
}


def load_labs(mimic_iv_dir: Path, target_hadm_ids: set) -> dict:
    """Load key lab results for target admissions (chunked read for 18GB file).

    Returns:
        labs_by_hadm: dict mapping hadm_id -> list of {label, value, unit, flag}
    """
    logger.info("Phase 5c: Loading lab events (chunked)...")

    lab_path = mimic_iv_dir / "hosp" / "labevents.csv"
    if not lab_path.exists():
        logger.warning(f"  Lab events not found at {lab_path}")
        return {}

    target_itemids = set(CXR_LAB_ITEMIDS.keys())

    # Collect raw lab rows, then pick the first per (hadm_id, itemid)
    raw_labs = {}  # (hadm_id, itemid) -> row dict
    chunk_size = 500000
    chunks_read = 0

    for chunk in pd.read_csv(lab_path, chunksize=chunk_size):
        chunks_read += 1
        # Filter to target admissions AND target lab items
        matched = chunk[
            (chunk["hadm_id"].isin(target_hadm_ids))
            & (chunk["itemid"].isin(target_itemids))
        ]
        for _, row in matched.iterrows():
            key = (int(row["hadm_id"]), int(row["itemid"]))
            if key not in raw_labs:
                raw_labs[key] = row
        if chunks_read % 50 == 0:
            logger.info(f"    Read {chunks_read} chunks, {len(raw_labs)} lab results so far...")

    # Organize by hadm_id
    labs_by_hadm = {}
    for (hadm_id, itemid), row in raw_labs.items():
        if hadm_id not in labs_by_hadm:
            labs_by_hadm[hadm_id] = []
        labs_by_hadm[hadm_id].append({
            "label": CXR_LAB_ITEMIDS.get(itemid, str(itemid)),
            "value": str(row["value"]) if pd.notna(row["value"]) else "",
            "unit": str(row["valueuom"]) if pd.notna(row["valueuom"]) else "",
            "flag": str(row["flag"]) if pd.notna(row["flag"]) else "normal",
        })

    logger.info(f"  Loaded labs for {len(labs_by_hadm)} admissions from {chunks_read} chunks")
    return labs_by_hadm


def compute_age(
    subject_id: int, study_date_str: str, patients_df: pd.DataFrame
) -> int | None:
    """Compute patient age at study time using MIMIC-IV anchor year."""
    pat = patients_df[patients_df["subject_id"] == subject_id]
    if pat.empty or not study_date_str:
        return None
    pat = pat.iloc[0]
    anchor_age = int(pat["anchor_age"])
    anchor_year = int(pat["anchor_year"])
    try:
        study_year = int(study_date_str[:4])
    except (ValueError, IndexError):
        return None
    return anchor_age + (study_year - anchor_year)


# ─── Phase 6: Assemble & Write JSON ────────────────────────────────────────


def assemble_entries(
    studies_df: pd.DataFrame,
    reports: dict,
    prior_index: dict,
    admissions_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    discharge_notes: dict,
    icd_by_hadm: dict,
    labs_by_hadm: dict,
    chexpert_df: pd.DataFrame | None,
    mimic_cxr_dir: Path,
) -> list[dict]:
    """Assemble enriched study entries."""
    logger.info("Phase 6: Assembling enriched entries...")

    chexpert_labels = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
        "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
        "Pneumothorax", "Support Devices",
    ]

    # Build chexpert lookup
    chexpert_lookup = {}
    if chexpert_df is not None:
        for _, row in chexpert_df.iterrows():
            stid = int(row["study_id"])
            labels = {}
            for label in chexpert_labels:
                val = row.get(label)
                labels[label] = float(val) if pd.notna(val) else None
            chexpert_lookup[stid] = labels

    entries = []
    stats = {
        "total": 0,
        "with_report": 0,
        "with_admission": 0,
        "with_discharge_note": 0,
        "with_priors": 0,
        "with_labs": 0,
    }

    for _, row in studies_df.iterrows():
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        dicom_id = str(row["dicom_id"])
        split = str(row["split"])
        study_date = convert_study_date(row["StudyDate"]) if pd.notna(row["StudyDate"]) else ""

        stats["total"] += 1

        # Image path
        sid_str = str(subject_id)
        stid_str = str(study_id)
        prefix = f"p{sid_str[:2]}"
        image_path = str(
            mimic_cxr_dir / "files" / prefix / f"p{sid_str}" / f"s{stid_str}" / f"{dicom_id}.jpg"
        )

        # Report
        rpt = reports.get(study_id)
        if not rpt:
            continue
        stats["with_report"] += 1
        report_gt = rpt["formatted"]

        # Prior studies
        priors = get_prior_studies(prior_index, subject_id, stid_str, study_date)
        if priors:
            stats["with_priors"] += 1

        # Admission linkage
        admission_info = None
        adm = match_study_to_admission(subject_id, study_date, admissions_df)
        if adm:
            stats["with_admission"] += 1
            hadm_id = int(adm["hadm_id"])

            # Demographics
            age = compute_age(subject_id, study_date, patients_df)
            pat = patients_df[patients_df["subject_id"] == subject_id]
            gender = str(pat.iloc[0]["gender"]) if not pat.empty else ""

            admission_info = {
                "hadm_id": hadm_id,
                "admittime": str(adm["admittime"]),
                "dischtime": str(adm["dischtime"]),
                "admission_type": str(adm["admission_type"]) if pd.notna(adm.get("admission_type")) else "",
                "demographics": {
                    "age": age,
                    "gender": gender,
                },
                "patient_history": "",
                "physical_examination": "",
                "chief_complaint": "",
                "medications_on_admission": "",
                "discharge_diagnosis": "",
                "icd_diagnoses": icd_by_hadm.get(hadm_id, []),
                "labs": labs_by_hadm.get(hadm_id, []),
            }

            if labs_by_hadm.get(hadm_id):
                stats["with_labs"] += 1

            # Extract sections from discharge note
            note = discharge_notes.get(hadm_id)
            if note:
                stats["with_discharge_note"] += 1
                sections = extract_admission_sections(note)
                admission_info["patient_history"] = sections["patient_history"]
                admission_info["physical_examination"] = sections["physical_examination"]
                admission_info["chief_complaint"] = sections["chief_complaint"]
                admission_info["medications_on_admission"] = sections["medications_on_admission"]
                admission_info["discharge_diagnosis"] = sections["discharge_diagnosis"]

        # CheXpert labels
        chexpert = chexpert_lookup.get(study_id)

        entry = {
            "study_id": stid_str,
            "subject_id": sid_str,
            "dicom_id": dicom_id,
            "split": split,
            "study_date": study_date,
            "image_path": image_path,
            "report_gt": report_gt,
            "admission_info": admission_info,
            "prior_studies": priors,
            "chexpert_labels": chexpert,
        }
        entries.append(entry)

    logger.info(f"  Assembled {len(entries)} entries")
    logger.info(f"  Stats: {stats}")
    return entries


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare enriched MIMIC-CXR validation & test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mimic_cxr_dir",
        required=True,
        help="MIMIC-CXR-JPEG root directory",
    )
    parser.add_argument(
        "--mimic_iv_dir",
        required=True,
        help="MIMIC-IV root directory (e.g., mimiciv/3.1)",
    )
    parser.add_argument(
        "--output",
        default="results/eval_enriched/",
        help="Output directory (default: results/eval_enriched/)",
    )
    parser.add_argument(
        "--splits",
        default="validate,test",
        help="Comma-separated splits to process (default: validate,test)",
    )
    parser.add_argument(
        "--skip_labs",
        action="store_true",
        help="Skip loading lab events (saves ~5 min on 18GB file)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mimic_cxr_dir = Path(args.mimic_cxr_dir)
    mimic_iv_dir = Path(args.mimic_iv_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",")]

    # Phase 1: Load MIMIC-CXR metadata
    studies_df, chexpert_df, split_df, meta_df = load_cxr_metadata(mimic_cxr_dir, splits)

    # Phase 2: Load CXR reports
    reports = load_reports(mimic_cxr_dir, studies_df)

    # Phase 3: Build prior study index (all splits for target subjects)
    target_subjects = set(studies_df["subject_id"].unique())
    prior_index = build_prior_index(mimic_cxr_dir, target_subjects, split_df, meta_df)

    # Phase 4: Link to admissions
    admissions_df, patients_df = link_admissions(mimic_iv_dir, studies_df)

    # Phase 5: Extract admission info

    # 5a: Match studies to admissions to find target hadm_ids
    logger.info("Matching studies to admissions...")
    target_hadm_ids = set()
    for _, row in studies_df.iterrows():
        subject_id = int(row["subject_id"])
        study_date = convert_study_date(row["StudyDate"]) if pd.notna(row["StudyDate"]) else ""
        adm = match_study_to_admission(subject_id, study_date, admissions_df)
        if adm:
            target_hadm_ids.add(int(adm["hadm_id"]))
    logger.info(f"  Matched {len(target_hadm_ids)} unique admissions")

    # 5b: Load discharge notes
    discharge_notes = load_discharge_notes(mimic_iv_dir, target_hadm_ids)

    # 5c: Load ICD diagnoses
    icd_by_hadm = load_icd_diagnoses(mimic_iv_dir, target_hadm_ids)

    # 5d: Load labs (optional)
    labs_by_hadm = {}
    if not args.skip_labs:
        labs_by_hadm = load_labs(mimic_iv_dir, target_hadm_ids)
    else:
        logger.info("Skipping lab events (--skip_labs)")

    # Phase 6: Assemble and write
    entries = assemble_entries(
        studies_df=studies_df,
        reports=reports,
        prior_index=prior_index,
        admissions_df=admissions_df,
        patients_df=patients_df,
        discharge_notes=discharge_notes,
        icd_by_hadm=icd_by_hadm,
        labs_by_hadm=labs_by_hadm,
        chexpert_df=chexpert_df,
        mimic_cxr_dir=mimic_cxr_dir,
    )

    # Write per-split output files
    split_names = {
        "validate": "val",
        "test": "test",
        "train": "train",
    }
    for split in splits:
        split_entries = [e for e in entries if e["split"] == split]
        if not split_entries:
            logger.warning(f"  No entries for split '{split}'")
            continue

        short = split_names.get(split, split)
        out_path = output_dir / f"{short}_studies_enriched.json"
        with open(out_path, "w") as f:
            json.dump(split_entries, f, indent=2, default=str)

        n_subjects = len(set(e["subject_id"] for e in split_entries))
        n_with_adm = sum(1 for e in split_entries if e["admission_info"] is not None)
        n_with_priors = sum(1 for e in split_entries if e["prior_studies"])
        avg_priors = (
            sum(len(e["prior_studies"]) for e in split_entries) / len(split_entries)
            if split_entries
            else 0
        )

        print(f"\n{split}: {len(split_entries)} studies, {n_subjects} subjects -> {out_path}")
        print(f"  Admission linked: {n_with_adm}/{len(split_entries)} ({100*n_with_adm/len(split_entries):.1f}%)")
        print(f"  With priors: {n_with_priors}/{len(split_entries)} ({100*n_with_priors/len(split_entries):.1f}%)")
        print(f"  Avg priors per study: {avg_priors:.1f}")

    # Also write combined file
    combined_path = output_dir / "all_studies_enriched.json"
    with open(combined_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)
    print(f"\nCombined: {len(entries)} studies -> {combined_path}")


if __name__ == "__main__":
    main()
