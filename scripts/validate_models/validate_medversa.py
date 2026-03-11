"""
Validate MedVersa (7B) inference.

Run on GPU server:
    python scripts/validate_models/validate_medversa.py --image path/to/cxr.jpg \
        --medversa_dir ../MedVersa

Requires MedVersa environment (see MedVersa repo for setup).
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


def validate(image_path: str, medversa_dir: str, device: str = "cuda"):
    print("=== Validating MedVersa (7B) ===")

    medversa_path = Path(medversa_dir)
    if not medversa_path.exists():
        print(f"ERROR: MedVersa dir not found: {medversa_path}")
        print("Clone: git clone https://huggingface.co/hyzhou/MedVersa")
        return

    sys.path.insert(0, str(medversa_path))

    try:
        from utils import registry, generate_predictions
        from torch import cuda
        print("MedVersa imports successful")
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Use the MedVersa conda environment: conda env create -f environment.yml")
        return

    # Load model
    print("Loading MedVersa model (7B, may take a minute)...")
    t0 = time.time()
    model_cls = registry.get_model_class("medomni")
    model = model_cls.from_pretrained("hyzhou/MedVersa").to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Run inference
    example = {
        "images": [image_path],
        "context": "Age:60-70.\nGender:M.\nIndication: Shortness of breath.",
        "prompt": "How would you characterize the findings from <img0>?",
        "modality": "cxr",
        "task": "report generation",
    }

    params = {
        "num_beams": 1,
        "do_sample": True,
        "min_length": 1,
        "top_p": 0.9,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "temperature": 0.1,
    }

    print("Generating report...")
    t0 = time.time()
    seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
        model,
        example["images"],
        example["context"],
        example["prompt"],
        example["modality"],
        example["task"],
        device=device,
        **params,
    )
    gen_time = time.time() - t0

    print(f"Generation time: {gen_time:.1f}s")
    print(f"\n--- Response ---\n{output_text}\n--- End ---\n")

    # Validate
    assert isinstance(output_text, str), "Output should be a string"
    assert len(output_text) > 20, f"Output too short ({len(output_text)} chars)"
    print("PASSED: MedVersa produced a report")

    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--medversa_dir", type=str,
                        default=str(Path(__file__).parent.parent.parent.parent / "MedVersa"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, args.medversa_dir, args.device)
