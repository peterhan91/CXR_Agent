"""
Validate FactCheXcker report verification pipeline.

Run on GPU server:
    python scripts/validate_models/validate_factchexcker.py --image path/to/cxr.jpg

Setup (one-time):
    git clone https://github.com/rajpurkarlab/FactCheXcker.git /path/to/FactCheXcker
    cd FactCheXcker && pip install -r requirements.txt
    pip install factchexcker-carinanet  # for CarinaNet component

NOTE: FactCheXcker requires an LLM backend (e.g., GPT-4 or Claude) for its
Query Generator and Code Generator stages. Configure in FactCheXcker/llm.py.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path


def download_sample_image():
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


def validate_carinanet(image_path: str):
    """Test the CarinaNet component independently."""
    print("=== Validating CarinaNet (FactCheXcker component) ===")
    try:
        import carinanet
        print("carinanet imported successfully")

        t0 = time.time()
        result = carinanet.predict_carina_ett(image_path)
        elapsed = time.time() - t0

        print(f"Inference time: {elapsed:.3f}s")
        print(f"Result keys: {list(result.keys())}")

        if "carina" in result:
            print(f"  Carina detected: {result['carina']}")
        if "ett" in result:
            print(f"  ETT detected: {result['ett']}")

        print("PASSED: CarinaNet produced predictions")
        return result

    except ImportError:
        print("CarinaNet not installed. Install with: pip install factchexcker-carinanet")
        return None
    except Exception as e:
        print(f"CarinaNet failed: {e}")
        return None


def validate_full_pipeline(image_path: str, factchexcker_dir: str):
    """Test the full FactCheXcker pipeline."""
    print("\n=== Validating FactCheXcker Full Pipeline ===")

    factchexcker_path = Path(factchexcker_dir)
    if not factchexcker_path.exists():
        print(f"ERROR: FactCheXcker dir not found: {factchexcker_path}")
        print("Clone: git clone https://github.com/rajpurkarlab/FactCheXcker.git")
        return

    sys.path.insert(0, str(factchexcker_path))

    try:
        from api import CXRModuleRegistry, CXRImage
        print("FactCheXcker API imported successfully")
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Install requirements: cd FactCheXcker && pip install -r requirements.txt")
        return

    # Test with a sample report containing a measurement
    sample_report = (
        "The endotracheal tube tip is positioned approximately 4.5 cm above the carina. "
        "Heart size is mildly enlarged. Bilateral pleural effusions are present, "
        "greater on the left. No pneumothorax."
    )

    config_path = factchexcker_path / "configs" / "config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        # List available configs
        configs = list(factchexcker_path.glob("configs/*.json"))
        print(f"Available configs: {configs}")
        return

    try:
        module_registry = CXRModuleRegistry(str(config_path))
        print("Module registry loaded")

        cxr_image = CXRImage(
            rid="test_001",
            image_path=image_path,
            report=sample_report,
            original_size=[1024, 1024],
            pixel_spacing=(0.139, 0.139),
            module_registry=module_registry,
        )
        print("CXRImage created")

        # Try the pipeline (this requires LLM access)
        from FactCheXcker import FactCheXcker as FactCheXckerPipeline
        pipeline = FactCheXckerPipeline(str(config_path))
        print("Pipeline instantiated")

        t0 = time.time()
        updated_report = pipeline.run_pipeline(cxr_image)
        elapsed = time.time() - t0

        print(f"\nPipeline time: {elapsed:.1f}s")
        print(f"\n--- Original Report ---\n{sample_report}")
        print(f"\n--- Updated Report ---\n{updated_report}\n")
        print("PASSED: Full pipeline produced updated report")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("\nNOTE: FactCheXcker requires an LLM backend configured in llm.py.")
        print("This is expected to fail without LLM configuration.")
        print("CarinaNet can still be used independently for ETT/carina detection.")


def validate(image_path: str, factchexcker_dir: str):
    # Test CarinaNet (standalone, no LLM needed)
    validate_carinanet(image_path)

    # Test full pipeline (requires LLM)
    validate_full_pipeline(image_path, factchexcker_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--factchexcker_dir", type=str,
                        default=str(Path(__file__).parent.parent.parent.parent / "FactCheXcker"))
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, args.factchexcker_dir)
