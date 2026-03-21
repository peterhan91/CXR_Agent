#!/usr/bin/env python3
"""
Rewrite HF dataset eval JSONs with absolute image paths for this machine.

After downloading the dataset from HuggingFace:
    huggingface-cli download peterhan91/CXR_Agent_Data --repo-type dataset --local-dir data/cxr_agent_dataset

Run this script to set up data/eval/ with working absolute paths:
    python scripts/setup_data.py
"""

import json
import glob
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "cxr_agent_dataset"
EVAL_SRC = DATASET_DIR / "eval"
EVAL_DST = REPO_ROOT / "data" / "eval"


def rewrite_paths(obj, base_dir: str):
    """Recursively rewrite relative image paths to absolute."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "path" in k.lower() and v.startswith("images/"):
                obj[k] = os.path.join(base_dir, v)
            elif isinstance(v, (dict, list)):
                rewrite_paths(v, base_dir)
    elif isinstance(obj, list):
        for item in obj:
            rewrite_paths(item, base_dir)


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print(
            "Run:\n"
            "  huggingface-cli download peterhan91/CXR_Agent_Data \\\n"
            "      --repo-type dataset \\\n"
            "      --local-dir data/cxr_agent_dataset"
        )
        return

    if not EVAL_SRC.exists():
        print(f"ERROR: No eval/ directory in dataset at {EVAL_SRC}")
        return

    EVAL_DST.mkdir(parents=True, exist_ok=True)
    base = str(DATASET_DIR)

    count = 0
    for jf in sorted(glob.glob(str(EVAL_SRC / "**/*.json"), recursive=True)):
        rel = os.path.relpath(jf, EVAL_SRC)
        with open(jf) as f:
            data = json.load(f)
        rewrite_paths(data, base)
        out = EVAL_DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {rel}")
        count += 1

    print(f"\nWrote {count} JSON files to {EVAL_DST}")
    print(f"Image paths now point to {DATASET_DIR}/images/")

    # Verify a few images exist
    sample_json = EVAL_DST / "mimic_cxr_test.json"
    if sample_json.exists():
        with open(sample_json) as f:
            studies = json.load(f)
        if studies:
            path = studies[0].get("image_path", "")
            exists = os.path.isfile(path)
            print(f"\nVerification: {path}")
            print(f"  Exists: {exists}")
            if not exists:
                print("  WARNING: Image not found. Check that the HF dataset downloaded correctly.")


if __name__ == "__main__":
    main()
