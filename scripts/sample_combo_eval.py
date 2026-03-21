#!/usr/bin/env python3
"""Sample stratified test sets for combo feature evaluation.

Creates two test sets from enriched data:
1. multiview_20.json — 20 studies WITH lateral views (for multiview eval)
2. temporal_20.json  — 20 studies WITH prior studies (for temporal eval)

Both are stratified across datasets where data is available.
All studies must have both FINDINGS and IMPRESSION in GT.
"""

import json
import os
import random

random.seed(42)

ENRICHED_DIR = "/home/than/DeepLearning/CXR_Agent/data/eval/enriched"
OUT_DIR = "/home/than/DeepLearning/CXR_Agent/data/eval/combo_test"


def has_valid_gt(s):
    """Study must have non-empty findings and impression."""
    findings = s.get("findings", "").strip()
    impression = s.get("impression", "").strip()
    return len(findings) > 20 and len(impression) > 5


def sample_multiview(n=20):
    """Sample studies with lateral views, stratified across datasets."""
    pools = {}
    for ds_file, ds_name in [
        ("mimic_cxr_test.json", "mimic_cxr"),
        ("rexgradient_test.json", "rexgradient"),
        ("chexpert_plus_valid.json", "chexpert_plus"),
        ("iu_xray_test.json", "iu_xray"),
    ]:
        with open(os.path.join(ENRICHED_DIR, ds_file)) as f:
            data = json.load(f)
        eligible = [s for s in data if s.get("lateral_image_path") and has_valid_gt(s)]
        random.shuffle(eligible)
        pools[ds_name] = eligible
        print(f"  {ds_name}: {len(eligible)} eligible (with lateral + valid GT)")

    # Stratified sampling: 5 per dataset where possible
    # MIMIC: 5, RexGradient: 5, IU: 5, CheXpert: 5 (only 10 available, take 5)
    samples = []
    allocation = {"mimic_cxr": 5, "rexgradient": 5, "iu_xray": 5, "chexpert_plus": 5}

    for ds_name, count in allocation.items():
        pool = pools[ds_name]
        take = min(count, len(pool))
        samples.extend(pool[:take])
        print(f"  -> sampled {take} from {ds_name}")

    print(f"  Total multiview samples: {len(samples)}")
    return samples


def sample_temporal(n=20):
    """Sample studies with prior studies, stratified across datasets."""
    pools = {}
    for ds_file, ds_name in [
        ("mimic_cxr_test.json", "mimic_cxr"),
        ("rexgradient_test.json", "rexgradient"),
    ]:
        with open(os.path.join(ENRICHED_DIR, ds_file)) as f:
            data = json.load(f)
        eligible = [
            s for s in data
            if s.get("prior_study") and s["prior_study"].get("image_path")
            and has_valid_gt(s)
        ]
        random.shuffle(eligible)
        pools[ds_name] = eligible
        print(f"  {ds_name}: {len(eligible)} eligible (with prior + valid GT)")

    # 10 MIMIC + 10 RexGradient
    samples = []
    allocation = {"mimic_cxr": 10, "rexgradient": 10}

    for ds_name, count in allocation.items():
        pool = pools[ds_name]
        take = min(count, len(pool))
        samples.extend(pool[:take])
        print(f"  -> sampled {take} from {ds_name}")

    print(f"  Total temporal samples: {len(samples)}")
    return samples


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Sampling multiview test set (studies with lateral views):")
    mv_samples = sample_multiview()
    mv_path = os.path.join(OUT_DIR, "multiview_20.json")
    with open(mv_path, "w") as f:
        json.dump(mv_samples, f, indent=2)
    print(f"  Saved: {mv_path}\n")

    print("Sampling temporal test set (studies with prior studies):")
    tp_samples = sample_temporal()
    tp_path = os.path.join(OUT_DIR, "temporal_20.json")
    with open(tp_path, "w") as f:
        json.dump(tp_samples, f, indent=2)
    print(f"  Saved: {tp_path}\n")

    # Print summary
    print("=== Summary ===")
    for name, samples in [("multiview", mv_samples), ("temporal", tp_samples)]:
        ds_counts = {}
        n_lat = sum(1 for s in samples if s.get("lateral_image_path"))
        n_prior = sum(1 for s in samples if s.get("prior_study") and s["prior_study"].get("image_path"))
        for s in samples:
            ds = s["dataset"]
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        ds_str = ", ".join(f"{d}={n}" for d, n in sorted(ds_counts.items()))
        print(f"  {name}: n={len(samples)}, lateral={n_lat}, prior={n_prior} ({ds_str})")


if __name__ == "__main__":
    main()
