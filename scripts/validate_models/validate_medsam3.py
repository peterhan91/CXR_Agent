"""
Validate MedSAM3 text-guided CXR segmentation.

Run on GPU server:
    python scripts/validate_models/validate_medsam3.py --image path/to/cxr.jpg

Setup (one-time):
    git clone https://github.com/Joey-S-Liu/MedSAM3.git /path/to/MedSAM3
    cd MedSAM3 && pip install -e .
    huggingface-cli login
"""

import argparse
import sys
import subprocess
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


CXR_PROMPTS = [
    "pleural effusion",
    "cardiomegaly",
    "lung opacity",
    "endotracheal tube",
]


def validate_cli(image_path: str, medsam3_dir: str, device: str = "cuda"):
    """Test MedSAM3 via its CLI interface."""
    print("=== Validating MedSAM3 (CLI mode) ===")

    medsam3_path = Path(medsam3_dir)
    infer_script = medsam3_path / "infer_sam.py"
    config_file = medsam3_path / "configs" / "full_lora_config.yaml"

    if not infer_script.exists():
        print(f"ERROR: infer_sam.py not found at {infer_script}")
        print("Clone MedSAM3: git clone https://github.com/Joey-S-Liu/MedSAM3.git")
        return

    for prompt_text in CXR_PROMPTS[:2]:  # Test 2 prompts
        output_path = tempfile.mktemp(suffix=".png")
        cmd = [
            sys.executable, str(infer_script),
            "--config", str(config_file),
            "--image", image_path,
            "--prompt", prompt_text,
            "--threshold", "0.5",
            "--nms-iou", "0.5",
            "--output", output_path,
        ]
        print(f"\nRunning: {prompt_text}")
        print(f"  Command: {' '.join(cmd)}")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  Time: {elapsed:.1f}s")
            if Path(output_path).exists():
                from PIL import Image
                mask = Image.open(output_path)
                print(f"  Output mask size: {mask.size}")
                print(f"  PASSED")
            else:
                print(f"  WARNING: Output file not created")
        else:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")


def validate_python(image_path: str, medsam3_dir: str, device: str = "cuda"):
    """Try to import MedSAM3 as a Python module for programmatic access."""
    print("\n=== Validating MedSAM3 (Python import) ===")

    medsam3_path = Path(medsam3_dir)
    sys.path.insert(0, str(medsam3_path))

    try:
        # Attempt to import the model — exact imports depend on MedSAM3's structure
        import yaml
        config_file = medsam3_path / "configs" / "full_lora_config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"Config loaded: {list(config.keys()) if config else 'empty'}")
        else:
            print(f"Config not found at {config_file}")

        # Try to import core modules
        try:
            from medsam3 import MedSAM3Model  # Speculative import
            print("Imported MedSAM3Model successfully")
        except ImportError:
            # Try alternative import paths
            try:
                from model import build_model
                print("Imported model.build_model successfully")
            except ImportError:
                print("NOTE: Could not find Python API for MedSAM3")
                print("MedSAM3 may only support CLI usage via infer_sam.py")
                print("Will use subprocess wrapper in tool implementation")

    except Exception as e:
        print(f"Python import failed: {e}")
        print("Will use CLI wrapper for MedSAM3 tool")


def validate(image_path: str, medsam3_dir: str, device: str = "cuda"):
    validate_python(image_path, medsam3_dir, device)
    validate_cli(image_path, medsam3_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--medsam3_dir", type=str,
                        default=str(Path(__file__).parent.parent.parent.parent / "MedSAM3"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, args.medsam3_dir, args.device)
