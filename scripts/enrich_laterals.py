#!/usr/bin/env python3
"""Enrich all 4 eval datasets with lateral_image_path.

Sources:
- MIMIC-CXR: mimic-cxr-2.0.0-metadata.csv (ViewPosition=LATERAL/LL)
- RexGradient: test_metadata_view_position.json (ImageViewPosition list)
- CheXpert+: view2_lateral.jpg in same study dir
- IU X-Ray: indiana_projections.csv (projection=Lateral)

Output: data/eval/enriched/<dataset>.json (same format, lateral_image_path added)
"""

import csv
import json
import os
import sys

MIMIC_CXR_DIR = "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0"
REXGRAD_VP_PATH = "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K/snapshots/b9277dd929190931f5a8b6ae19f3e3c8aae2f1f7/metadata/test_metadata_view_position.json"
REXGRAD_IMG_BASE = "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K/snapshots/e0c7dd5940c6e5f77aac20eea3ff93825d7f8ff3/images/deid_png"
CHEXPERT_BASE = "/home/than/Datasets/stanford_mit_chest/CheXpert-v1.0/valid"
IU_PROJ_CSV = "/home/than/Datasets/IU_XRay/indiana_projections.csv"
IU_IMG_DIR = "/home/than/Datasets/IU_XRay/images/images_normalized"

DATA_DIR = "/home/than/DeepLearning/CXR_Agent/data/eval"
OUT_DIR = os.path.join(DATA_DIR, "enriched")


def enrich_mimic():
    """Add lateral_image_path to MIMIC-CXR test set."""
    print("=== MIMIC-CXR ===")
    # Build study_id -> lateral image path from metadata CSV
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
                MIMIC_CXR_DIR, "files", p_prefix, f"p{subj}",
                f"s{sid}", f"{row['dicom_id']}.jpg",
            )
            laterals[sid] = img_path

    with open(os.path.join(DATA_DIR, "mimic_cxr_test.json")) as f:
        data = json.load(f)

    enriched = 0
    for entry in data:
        sid_num = entry["study_id"].replace("mimic_", "")
        lat = laterals.get(sid_num)
        if lat and os.path.exists(lat):
            entry["lateral_image_path"] = lat
            enriched += 1
        else:
            entry["lateral_image_path"] = None

    out_path = os.path.join(OUT_DIR, "mimic_cxr_test.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {enriched}/{len(data)} enriched with lateral -> {out_path}")
    return data


def enrich_rexgradient():
    """Add lateral_image_path to RexGradient test set."""
    print("=== RexGradient ===")
    with open(REXGRAD_VP_PATH) as f:
        vp_data = json.load(f)

    # Build key -> lateral abs path
    laterals = {}
    for key, entry in vp_data.items():
        views = entry.get("ImageViewPosition", [])
        paths = entry.get("ImagePath", [])
        lateral_idx = None
        for i, v in enumerate(views):
            if v in ("LATERAL", "LL", "LAT"):
                lateral_idx = i
                break
        if lateral_idx is not None and lateral_idx < len(paths):
            rel_path = paths[lateral_idx]
            # Strip leading ../deid_png/
            if rel_path.startswith("../deid_png/"):
                rel_path = rel_path[len("../deid_png/"):]
            elif rel_path.startswith("../"):
                rel_path = rel_path[3:]
                if rel_path.startswith("deid_png/"):
                    rel_path = rel_path[9:]
            abs_path = os.path.join(REXGRAD_IMG_BASE, rel_path)
            laterals[key] = abs_path

    with open(os.path.join(DATA_DIR, "rexgradient_test.json")) as f:
        data = json.load(f)

    enriched = 0
    for entry in data:
        vp_key = entry["study_id"].replace("rexgrad_", "")
        lat = laterals.get(vp_key)
        if lat and os.path.exists(lat):
            entry["lateral_image_path"] = lat
            enriched += 1
        else:
            entry["lateral_image_path"] = None

    out_path = os.path.join(OUT_DIR, "rexgradient_test.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {enriched}/{len(data)} enriched with lateral -> {out_path}")
    return data


def enrich_chexpert():
    """Add lateral_image_path to CheXpert+ valid set."""
    print("=== CheXpert+ ===")
    with open(os.path.join(DATA_DIR, "chexpert_plus_valid.json")) as f:
        data = json.load(f)

    enriched = 0
    for entry in data:
        img_path = entry["image_path"]
        study_dir = os.path.dirname(img_path)
        # Look for view*_lateral.jpg in same directory
        lat_path = None
        if os.path.exists(study_dir):
            for fname in sorted(os.listdir(study_dir)):
                if "lateral" in fname.lower() and fname.endswith(".jpg"):
                    lat_path = os.path.join(study_dir, fname)
                    break
        if lat_path and os.path.exists(lat_path):
            entry["lateral_image_path"] = lat_path
            enriched += 1
        else:
            entry["lateral_image_path"] = None

    out_path = os.path.join(OUT_DIR, "chexpert_plus_valid.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {enriched}/{len(data)} enriched with lateral -> {out_path}")
    return data


def enrich_iu_xray():
    """Add lateral_image_path to IU X-Ray test set."""
    print("=== IU X-Ray ===")
    # Build uid -> lateral filename from projections CSV
    laterals_by_uid = {}
    with open(IU_PROJ_CSV) as f:
        for row in csv.DictReader(f):
            if row["projection"] == "Lateral":
                laterals_by_uid[row["uid"]] = row["filename"]

    with open(os.path.join(DATA_DIR, "iu_xray_test.json")) as f:
        data = json.load(f)

    enriched = 0
    for entry in data:
        # study_id format: iu_CXR{uid}_IM-...
        sid = entry["study_id"]
        # Extract uid from image filename: {uid}_IM-...
        img_basename = os.path.basename(entry["image_path"])
        uid = img_basename.split("_")[0]

        lat_fname = laterals_by_uid.get(uid)
        if lat_fname:
            lat_path = os.path.join(IU_IMG_DIR, lat_fname)
            if os.path.exists(lat_path):
                entry["lateral_image_path"] = lat_path
                enriched += 1
            else:
                entry["lateral_image_path"] = None
        else:
            entry["lateral_image_path"] = None

    out_path = os.path.join(OUT_DIR, "iu_xray_test.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {enriched}/{len(data)} enriched with lateral -> {out_path}")
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mimic = enrich_mimic()
    rexgrad = enrich_rexgradient()
    chexpert = enrich_chexpert()
    iu = enrich_iu_xray()

    # Print summary
    print("\n=== Summary ===")
    for name, data in [("mimic_cxr", mimic), ("rexgradient", rexgrad),
                        ("chexpert_plus", chexpert), ("iu_xray", iu)]:
        n_lat = sum(1 for s in data if s.get("lateral_image_path"))
        n_prior = sum(1 for s in data if s.get("prior_study") and s["prior_study"].get("image_path"))
        print(f"  {name:<16} n={len(data):>5}  lateral={n_lat:>5}  prior={n_prior:>5}")


if __name__ == "__main__":
    main()
