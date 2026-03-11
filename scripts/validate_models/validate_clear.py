"""
Validate CLEAR concept scoring on a single CXR image.

Run on GPU server:
    python scripts/validate_models/validate_clear.py --image path/to/cxr.jpg

Requires:
    - CLEAR model checkpoint at ../cxr_concept/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt
    - Concepts file at ../cxr_concept/concepts/mimic_concepts.csv
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clear.concept_scorer import CLEARConceptScorer


def download_sample_image():
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


def validate(image_path: str, model_path: str = None, concepts_path: str = None,
             device: str = "cuda"):
    print("=== Validating CLEAR Concept Scorer ===")

    # Step 1: Initialize
    scorer = CLEARConceptScorer(
        model_path=model_path,
        concepts_path=concepts_path,
        device=device,
    )
    print(f"Model path: {scorer.model_path}")
    print(f"Concepts path: {scorer.concepts_path}")

    # Check files exist
    assert Path(scorer.model_path).exists(), f"Model not found: {scorer.model_path}"
    assert Path(scorer.concepts_path).exists(), f"Concepts not found: {scorer.concepts_path}"

    # Step 2: Load model
    print("Loading CLEAR model and encoding concepts...")
    t0 = time.time()
    scorer.load()
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Concepts: {len(scorer.concepts)}")
    print(f"Concept features shape: {scorer.concept_features.shape}")

    # Step 3: Score single image
    print(f"\nScoring image: {image_path}")
    t0 = time.time()
    prior_text = scorer.score_image(image_path, top_k=20)
    score_time = time.time() - t0
    print(f"Scoring time: {score_time:.3f}s")
    print(f"\n--- Concept Prior ---\n{prior_text}\n--- End ---\n")

    # Step 4: Also test raw scores
    raw_scores = scorer.score_image_raw(image_path)
    print(f"Raw scores shape: {raw_scores.shape}")
    print(f"Score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    print(f"Mean: {raw_scores.mean():.4f}, Std: {raw_scores.std():.4f}")

    # Step 5: Validate
    assert raw_scores.shape[0] == len(scorer.concepts), "Score count mismatch"
    assert raw_scores.max() > 0.05, "All scores near zero — model may not be working"
    assert "## CLEAR Concept Prior" in prior_text, "Missing template header"
    print("\nPASSED: Scores computed, format correct, values reasonable")

    return prior_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--concepts_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, args.model_path, args.concepts_path, args.device)
