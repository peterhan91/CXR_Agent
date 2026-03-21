#!/usr/bin/env python3
"""Sample ~20 stratified studies for feature testing.

For each dataset (5 per dataset), prefer studies that have:
1. Both FINDINGS and IMPRESSION in GT (required for all)
2. At least some with prior_study (image_path + report) — for Features 2, 3
3. At least some with lateral_image_path — for Feature 4
4. At least some with metadata (age, sex, indication) — for Feature 5

MIMIC-CXR metadata: entry["metadata"] has subject_id but age/sex require
  MIMIC-IV patients.csv join. Uses enriched admission_info.demographics if
  available, otherwise computes from patients.csv anchor_age + study_year.

Output: data/eval/feature_test/all_20.json
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

MIMIC_CXR_DIR = "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0"
MIMIC_IV_DIR = "/home/than/physionet.org/files/mimiciv/3.1"
DATA_DIR = "data/eval"
OUTPUT_DIR = "data/eval/feature_test"


def load_mimic_patients():
    """Load MIMIC-IV patients.csv for age/sex lookup by subject_id."""
    patients = {}
    path = os.path.join(MIMIC_IV_DIR, "hosp", "patients.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(MIMIC_IV_DIR, "hosp", "patients.csv")
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for row in csv.DictReader(f):
            patients[row["subject_id"]] = {
                "gender": row["gender"],
                "anchor_age": int(row["anchor_age"]),
                "anchor_year": int(row["anchor_year"]),
            }
    print(f"  Loaded {len(patients)} patients from MIMIC-IV")
    return patients


def load_mimic_laterals():
    """Build study_id -> lateral_image_path from MIMIC-CXR metadata CSV."""
    # Need split map to construct file paths
    split_map = {}
    with open(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-split.csv")) as f:
        for row in csv.DictReader(f):
            split_map[row["study_id"]] = row["subject_id"]

    laterals = {}
    with open(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-metadata.csv")) as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            view = row.get("ViewPosition", "")
            if view not in ("LATERAL", "LL") or sid in laterals:
                continue
            if sid not in split_map:
                continue
            subj = split_map[sid]
            p_prefix = f"p{subj[:2]}"
            img_path = os.path.join(
                MIMIC_CXR_DIR,
                "files",
                p_prefix,
                f"p{subj}",
                f"s{sid}",
                f"{row['dicom_id']}.jpg",
            )
            laterals[sid] = img_path
    print(f"  Found {len(laterals)} MIMIC-CXR studies with lateral views")
    return laterals


def enrich_mimic_entry(entry, patients, laterals):
    """Add age, sex, indication, lateral_image_path to MIMIC-CXR entry."""
    meta = entry["metadata"]
    subj_id = str(meta.get("subject_id", ""))

    # Age/sex: try enriched admission_info first, then MIMIC-IV patients.csv
    age, sex = None, None
    ai = meta.get("admission_info")
    if ai and isinstance(ai, dict):
        demo = ai.get("demographics", {})
        if demo:
            age = demo.get("age")
            sex = demo.get("gender")

    if (age is None or sex is None) and subj_id in patients:
        p = patients[subj_id]
        study_date = meta.get("study_date", "")
        if study_date and len(study_date) >= 4:
            study_year = int(study_date[:4])
            age = p["anchor_age"] + (study_year - p["anchor_year"])
        sex = p["gender"]

    meta["age"] = age
    meta["sex"] = sex

    # Indication from enriched admission_info (patient_history or chief_complaint)
    indication = ""
    if ai and isinstance(ai, dict):
        indication = (ai.get("patient_history") or ai.get("chief_complaint") or "").strip()
    meta["indication"] = indication

    # Lateral view
    sid_num = entry["study_id"].replace("mimic_", "")
    lateral = laterals.get(sid_num, "")
    if lateral and os.path.exists(lateral):
        entry["lateral_image_path"] = lateral
    else:
        entry["lateral_image_path"] = ""

    return entry


def has_gt(entry):
    """Check if entry has both FINDINGS and IMPRESSION ground truth."""
    return bool(entry.get("findings", "").strip()) and bool(
        entry.get("impression", "").strip()
    )


def has_metadata(entry):
    """Check if entry has age and sex."""
    meta = entry.get("metadata", {})
    return meta.get("age") is not None and bool(meta.get("sex"))


def has_indication(entry):
    """Check if entry has indication text."""
    meta = entry.get("metadata", {})
    return bool((meta.get("indication") or "").strip())


def has_prior(entry):
    """Check if entry has prior study with image path."""
    ps = entry.get("prior_study")
    return ps is not None and bool(ps.get("image_path"))


def has_lateral(entry):
    """Check if entry has lateral image path."""
    return bool(entry.get("lateral_image_path", ""))


def sample_from_pool(pool, n, priority_fns=None):
    """Sample n entries from pool, prioritizing entries matching priority_fns.

    priority_fns: list of (fn, min_count) tuples. Try to satisfy each minimum.
    """
    if len(pool) <= n:
        return pool[:]

    if not priority_fns:
        random.shuffle(pool)
        return pool[:n]

    selected = []
    remaining = pool[:]

    # Satisfy priority requirements first
    for fn, min_count in priority_fns:
        matching = [e for e in remaining if fn(e) and e not in selected]
        random.shuffle(matching)
        needed = min(min_count, n - len(selected))
        for e in matching[:needed]:
            selected.append(e)
            remaining.remove(e)

    # Fill remaining slots
    if len(selected) < n:
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


def main():
    parser = argparse.ArgumentParser(description="Sample stratified eval set for feature testing")
    parser.add_argument("--per-dataset", type=int, default=5, help="Studies per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "all_20.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    n = args.per_dataset

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── Load MIMIC-IV reference data ──
    print("Loading MIMIC-IV reference data...")
    patients = load_mimic_patients()
    laterals = load_mimic_laterals()

    all_sampled = []
    stats = {}

    # ── Dataset 1: MIMIC-CXR ──
    print("\n=== MIMIC-CXR ===")
    with open(os.path.join(DATA_DIR, "mimic_cxr_test.json")) as f:
        mimic = json.load(f)
    print(f"  Loaded {len(mimic)} entries")

    # Enrich with age/sex/indication/lateral
    for e in mimic:
        enrich_mimic_entry(e, patients, laterals)

    # Filter: must have FINDINGS + IMPRESSION
    pool = [e for e in mimic if has_gt(e)]
    print(f"  With FINDINGS+IMPRESSION: {len(pool)}")

    # Sample: prioritize studies with lateral + prior + metadata
    sampled = sample_from_pool(
        pool,
        n,
        priority_fns=[
            (lambda e: has_lateral(e) and has_prior(e) and has_metadata(e), 3),
            (has_lateral, 2),
            (has_prior, 2),
        ],
    )
    all_sampled.extend(sampled)

    n_meta = sum(1 for e in sampled if has_metadata(e))
    n_prior = sum(1 for e in sampled if has_prior(e))
    n_lat = sum(1 for e in sampled if has_lateral(e))
    n_ind = sum(1 for e in sampled if has_indication(e))
    stats["mimic_cxr"] = {"total": len(sampled), "metadata": n_meta, "prior": n_prior, "lateral": n_lat, "indication": n_ind}
    print(f"  Sampled {len(sampled)}: {n_meta} metadata, {n_prior} prior, {n_lat} lateral, {n_ind} indication")

    # ── Dataset 2: ReXGradient ──
    print("\n=== ReXGradient ===")
    with open(os.path.join(DATA_DIR, "rexgradient_test.json")) as f:
        rexgrad = json.load(f)
    print(f"  Loaded {len(rexgrad)} entries")

    # ReXGradient has age/sex/indication in metadata already
    pool = [e for e in rexgrad if has_gt(e)]
    print(f"  With FINDINGS+IMPRESSION: {len(pool)}")

    sampled = sample_from_pool(
        pool,
        n,
        priority_fns=[
            (lambda e: has_prior(e) and has_metadata(e), 3),
            (has_prior, 2),
        ],
    )
    all_sampled.extend(sampled)

    n_meta = sum(1 for e in sampled if has_metadata(e))
    n_prior = sum(1 for e in sampled if has_prior(e))
    n_lat = sum(1 for e in sampled if has_lateral(e))
    n_ind = sum(1 for e in sampled if has_indication(e))
    stats["rexgradient"] = {"total": len(sampled), "metadata": n_meta, "prior": n_prior, "lateral": n_lat, "indication": n_ind}
    print(f"  Sampled {len(sampled)}: {n_meta} metadata, {n_prior} prior, {n_lat} lateral, {n_ind} indication")

    # ── Dataset 3: CheXpert-Plus ──
    print("\n=== CheXpert-Plus ===")
    with open(os.path.join(DATA_DIR, "chexpert_plus_valid.json")) as f:
        chexpert = json.load(f)
    print(f"  Loaded {len(chexpert)} entries")

    pool = [e for e in chexpert if has_gt(e)]
    print(f"  With FINDINGS+IMPRESSION: {len(pool)}")

    sampled = sample_from_pool(pool, n)
    all_sampled.extend(sampled)

    n_meta = sum(1 for e in sampled if has_metadata(e))
    stats["chexpert_plus"] = {"total": len(sampled), "metadata": n_meta, "prior": 0, "lateral": 0, "indication": 0}
    print(f"  Sampled {len(sampled)}: {n_meta} metadata")

    # ── Dataset 4: IU-Xray ──
    print("\n=== IU-Xray ===")
    with open(os.path.join(DATA_DIR, "iu_xray_test.json")) as f:
        iu = json.load(f)
    print(f"  Loaded {len(iu)} entries")

    pool = [e for e in iu if has_gt(e)]
    print(f"  With FINDINGS+IMPRESSION: {len(pool)}")

    sampled = sample_from_pool(pool, n)
    all_sampled.extend(sampled)

    n_ind = sum(1 for e in sampled if has_indication(e))
    stats["iu_xray"] = {"total": len(sampled), "metadata": 0, "prior": 0, "lateral": 0, "indication": n_ind}
    print(f"  Sampled {len(sampled)}: {n_ind} with indication")

    # ── Summary ──
    total = len(all_sampled)
    total_meta = sum(s["metadata"] for s in stats.values())
    total_prior = sum(s["prior"] for s in stats.values())
    total_lateral = sum(s["lateral"] for s in stats.values())
    total_ind = sum(s["indication"] for s in stats.values())

    print(f"\n{'='*50}")
    print(f"Total sampled: {total}")
    print(f"  With metadata (age+sex): {total_meta} (req >=15)")
    print(f"  With prior study:        {total_prior} (req >=5)")
    print(f"  With lateral view:       {total_lateral} (req >=5)")
    print(f"  With indication:         {total_ind}")

    # Verify requirements
    ok = True
    if total_meta < 15:
        print(f"  WARNING: metadata count {total_meta} < 15 requirement")
        ok = False
    if total_prior < 5:
        print(f"  WARNING: prior count {total_prior} < 5 requirement")
        ok = False
    if total_lateral < 5:
        print(f"  WARNING: lateral count {total_lateral} < 5 requirement")
        ok = False

    if ok:
        print("  All requirements met!")

    # ── Write output ──
    with open(args.output, "w") as f:
        json.dump(all_sampled, f, indent=2)
    print(f"\nWritten to {args.output}")

    # ── Write stats ──
    stats_path = os.path.join(os.path.dirname(args.output), "sample_stats.json")
    stats["_summary"] = {
        "total": total,
        "with_metadata": total_meta,
        "with_prior": total_prior,
        "with_lateral": total_lateral,
        "with_indication": total_ind,
        "seed": args.seed,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
