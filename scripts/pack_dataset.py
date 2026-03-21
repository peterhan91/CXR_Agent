#!/usr/bin/env python3
"""Pack all eval data + referenced CXR images into a HuggingFace-uploadable dataset."""

import json
import glob
import os
import shutil
from pathlib import Path
from collections import defaultdict

EVAL_DIR = Path("/home/than/DeepLearning/CXR_Agent/data/eval")
OUT_DIR  = Path("/home/than/DeepLearning/CXR_Agent/data/cxr_agent_dataset")

# Prefix stripping rules: old_prefix -> new_prefix
PREFIX_MAP = {
    "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/files/":    "images/mimic/",
    "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K/snapshots/e0c7dd5940c6e5f77aac20eea3ff93825d7f8ff3/images/deid_png/": "images/rexgradient/",
    "/home/than/physionet.org/files/chexpert-plus/1.0.0/PNG/":      "images/chexpert/",
    "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K/snapshots/": "images/rexgradient_other/",
}
# IU X-ray and other potential paths - discover dynamically
IU_PREFIXES = [
    "/home/than/physionet.org/files/",
    "/home/than/.cache/huggingface/",
    "/home/than/DeepLearning/",
]


def remap_path(old_path: str) -> str:
    """Convert absolute path to dataset-relative path."""
    for old_prefix, new_prefix in PREFIX_MAP.items():
        if old_path.startswith(old_prefix):
            return new_prefix + old_path[len(old_prefix):]
    # Fallback: use last 4 path components
    parts = old_path.split("/")
    return "images/other/" + "/".join(parts[-4:])


def collect_and_rewrite(obj, path_map: dict):
    """Recursively find path-like keys, collect old->new mapping, rewrite in-place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "path" in k.lower() and "/" in v:
                new_path = remap_path(v)
                path_map[v] = new_path
                obj[k] = new_path
            elif isinstance(v, (dict, list)):
                collect_and_rewrite(v, path_map)
    elif isinstance(obj, list):
        for item in obj:
            collect_and_rewrite(item, path_map)


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # 1. Discover all eval JSONs
    json_files = sorted(glob.glob(str(EVAL_DIR / "**/*.json"), recursive=True))
    print(f"Found {len(json_files)} JSON files")

    # 2. Collect all image paths and rewrite JSONs
    path_map = {}  # old_abs_path -> new_relative_path
    for jf in json_files:
        rel = os.path.relpath(jf, EVAL_DIR)
        try:
            data = json.load(open(jf))
        except Exception as e:
            print(f"  SKIP {rel}: {e}")
            continue
        collect_and_rewrite(data, path_map)
        # Write rewritten JSON
        out_json = OUT_DIR / "eval" / rel
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Rewrote {rel}")

    print(f"\nTotal unique image paths: {len(path_map)}")

    # 3. Copy images
    copied = 0
    missing = 0
    total = len(path_map)
    by_dataset = defaultdict(int)
    for i, (old_path, new_path) in enumerate(sorted(path_map.items())):
        dataset = new_path.split("/")[1] if "/" in new_path else "unknown"
        dst = OUT_DIR / new_path
        if not os.path.exists(old_path):
            print(f"  MISSING: {old_path}")
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, dst)
        copied += 1
        by_dataset[dataset] += 1
        if (i + 1) % 1000 == 0:
            print(f"  Copied {i+1}/{total} images...")

    print(f"\nCopied {copied} images ({missing} missing)")
    for ds, cnt in sorted(by_dataset.items()):
        print(f"  {ds}: {cnt}")

    # 4. Save path mapping for reference
    with open(OUT_DIR / "path_mapping.json", "w") as f:
        json.dump(path_map, f, indent=2)

    # 5. Print total size
    total_size = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    )
    print(f"\nDataset dir: {OUT_DIR}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")
    print("\nDone! Upload with:")
    print(f"  huggingface-cli upload peterhan91/CXR_Agent_Data {OUT_DIR} --repo-type dataset")


if __name__ == "__main__":
    main()
