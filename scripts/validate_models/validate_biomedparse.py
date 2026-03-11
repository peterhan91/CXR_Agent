"""
Validate BiomedParse v1 for 2D CXR segmentation.

Run on GPU server:
    python scripts/validate_models/validate_biomedparse.py --image path/to/cxr.jpg

Setup (one-time):
    git clone https://github.com/microsoft/BiomedParse.git /path/to/BiomedParse
    cd /path/to/BiomedParse
    pip install -r assets/requirements/requirements.txt
    pip install git+https://github.com/facebookresearch/detectron2.git

    Or install from HuggingFace:
    pip install huggingface_hub
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def download_sample_image():
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


# Verified CXR text prompts for BiomedParse v1
CXR_PROMPTS = [
    "left lung",
    "right lung",
    "lung opacity",
    "viral pneumonia",
    "COVID-19 infection",
]


def validate(image_path: str, biomedparse_dir: str = None, device: str = "cuda"):
    """Test BiomedParse segmentation on a CXR image."""
    print("=== Validating BiomedParse v1 (2D CXR segmentation) ===")

    # Try import from installed package or local clone
    if biomedparse_dir:
        sys.path.insert(0, biomedparse_dir)

    try:
        from PIL import Image
        from modeling.BaseModel import BaseModel
        from modeling import build_model
        from utilities.distributed import init_distributed
        from utilities.arguments import load_opt_from_config_files
        from inference_utils.inference import interactive_infer_image
        print("BiomedParse imports successful")
    except ImportError as e:
        print(f"ERROR: Cannot import BiomedParse: {e}")
        print("Install BiomedParse first:")
        print("  git clone https://github.com/microsoft/BiomedParse.git")
        print("  cd BiomedParse && pip install -r assets/requirements/requirements.txt")
        return

    # Load model
    print("Loading BiomedParse model...")
    t0 = time.time()

    config_path = "configs/biomedparse_inference.yaml"
    if biomedparse_dir:
        config_path = str(Path(biomedparse_dir) / config_path)

    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(
        "hf_hub:microsoft/BiomedParse"
    ).eval().cuda()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Test each CXR prompt
    for prompt_text in CXR_PROMPTS:
        print(f"\nSegmenting: '{prompt_text}'")
        t0 = time.time()
        pred_mask = interactive_infer_image(model, image, [prompt_text])
        seg_time = time.time() - t0

        mask = pred_mask[0]  # First prompt result
        print(f"  Time: {seg_time:.3f}s")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Value range: [{mask.min():.4f}, {mask.max():.4f}]")

        # Threshold to binary
        binary_mask = (mask > 0.5).astype(np.float32)
        coverage = binary_mask.sum() / binary_mask.size * 100
        print(f"  Coverage: {coverage:.1f}% of image")

        # Basic validation
        assert mask.shape[0] > 100 and mask.shape[1] > 100, "Mask too small"

    print("\nPASSED: All CXR prompts produced segmentation masks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--biomedparse_dir", type=str, default=None,
                        help="Path to cloned BiomedParse repo")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, args.biomedparse_dir, args.device)
