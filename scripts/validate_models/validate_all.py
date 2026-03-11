"""
Run all model validation scripts sequentially.

Usage:
    python scripts/validate_models/validate_all.py --image path/to/cxr.jpg
    python scripts/validate_models/validate_all.py  # downloads sample image

This will test each model independently. Failures in one model
won't prevent testing of others.

GPU allocation suggestion for 3x A6000 (48GB each):
    GPU 0: CheXagent-2 (~6GB) + CLEAR (~2GB)
    GPU 1: CheXOne (~6GB) + BiomedParse (~4GB)
    GPU 2: MedVersa (~14GB) + MedSAM3 (~4GB) + FactCheXcker (~2GB)
"""

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def download_sample_image():
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


def run_validation(name, func, *args, **kwargs):
    """Run a single validation with error handling."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        return True, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FAIL] {name} failed after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip: chexagent2 chexone clear medversa biomedparse medsam3 factchexcker")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these models")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
    print(f"Test image: {args.image}\n")

    project_root = Path(__file__).parent.parent.parent
    parent_dir = project_root.parent  # ../

    results = {}
    models_to_run = args.only or [
        "chexagent2", "chexone", "clear",
        "medversa", "biomedparse", "medsam3", "factchexcker",
    ]
    models_to_run = [m for m in models_to_run if m not in args.skip]

    # --- CheXagent-2 ---
    if "chexagent2" in models_to_run:
        from validate_chexagent2 import validate as validate_chexagent2
        ok, _ = run_validation(
            "CheXagent-2-3b (report generation)",
            validate_chexagent2, args.image, srrg=False,
        )
        results["chexagent2"] = ok

        ok, _ = run_validation(
            "CheXagent-2-3b-SRRG (structured findings)",
            validate_chexagent2, args.image, srrg=True,
        )
        results["chexagent2_srrg"] = ok

    # --- CheXOne ---
    if "chexone" in models_to_run:
        from validate_chexone import validate as validate_chexone
        ok, _ = run_validation(
            "CheXOne (Qwen2.5-VL report generation)",
            validate_chexone, args.image,
        )
        results["chexone"] = ok

    # --- CLEAR ---
    if "clear" in models_to_run:
        from validate_clear import validate as validate_clear
        ok, _ = run_validation(
            "CLEAR (concept scoring)",
            validate_clear, args.image,
        )
        results["clear"] = ok

    # --- MedVersa ---
    if "medversa" in models_to_run:
        from validate_medversa import validate as validate_medversa
        ok, _ = run_validation(
            "MedVersa (7B report generation)",
            validate_medversa, args.image,
            str(parent_dir / "MedVersa"),
        )
        results["medversa"] = ok

    # --- BiomedParse ---
    if "biomedparse" in models_to_run:
        from validate_biomedparse import validate as validate_biomedparse
        ok, _ = run_validation(
            "BiomedParse v1 (CXR segmentation)",
            validate_biomedparse, args.image,
            str(parent_dir / "BiomedParse"),
        )
        results["biomedparse"] = ok

    # --- MedSAM3 ---
    if "medsam3" in models_to_run:
        from validate_medsam3 import validate as validate_medsam3
        ok, _ = run_validation(
            "MedSAM3 (text-guided segmentation)",
            validate_medsam3, args.image,
            str(parent_dir / "MedSAM3"),
        )
        results["medsam3"] = ok

    # --- FactCheXcker ---
    if "factchexcker" in models_to_run:
        from validate_factchexcker import validate as validate_factchexcker
        ok, _ = run_validation(
            "FactCheXcker (report verification)",
            validate_factchexcker, args.image,
            str(parent_dir / "FactCheXcker"),
        )
        results["factchexcker"] = ok

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}\n")

    for model, passed in results.items():
        status = "[OK]  " if passed else "[FAIL]"
        print(f"  {status} {model}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} models validated successfully")

    if passed < total:
        print("\n  Failed models may need:")
        print("  - Dependencies installed (see each script header)")
        print("  - Model weights downloaded")
        print("  - External repos cloned (BiomedParse, MedSAM3, FactCheXcker)")


if __name__ == "__main__":
    # Add the validate_models dir to path so individual scripts can be imported
    sys.path.insert(0, str(Path(__file__).parent))
    main()
