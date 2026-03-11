"""
Validate CheXagent-2-3b and CheXagent-2-3b-srrg-findings inference.

Run on GPU server:
    python scripts/validate_models/validate_chexagent2.py --image path/to/cxr.jpg
    python scripts/validate_models/validate_chexagent2.py --image path/to/cxr.jpg --srrg

If no --image provided, downloads a sample CXR from HuggingFace.
"""

import argparse
import tempfile
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_sample_image():
    """Download a sample CXR image for testing."""
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


def validate(image_path: str, srrg: bool = False, device: str = "cuda"):
    model_name = (
        "StanfordAIMI/CheXagent-2-3b-srrg-findings" if srrg
        else "StanfordAIMI/CheXagent-2-3b"
    )
    print(f"=== Validating {model_name} ===")

    # Step 1: Load
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model = model.to(torch.bfloat16)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Step 2: Prepare input
    paths = [image_path]
    prompt = (
        "Structured Radiology Report Generation for Findings Section" if srrg
        else "Generate a radiology report for this chest X-ray."
    )

    query = tokenizer.from_list_format(
        [*[{"image": path} for path in paths], {"text": prompt}]
    )
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]

    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    )
    print(f"Input tokens: {input_ids.shape[1]}")

    # Step 3: Generate
    print("Generating...")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids.to(device),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=512,
        )[0]
    gen_time = time.time() - t0

    response = tokenizer.decode(output[input_ids.size(1):-1])
    print(f"Generation time: {gen_time:.1f}s")
    print(f"Output tokens: {len(output) - input_ids.size(1)}")
    print(f"\n--- Response ---\n{response}\n--- End ---\n")

    # Step 4: Validate output format
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 20, f"Response too short ({len(response)} chars)"
    print("PASSED: Response is a non-trivial string")

    if srrg:
        # SRRG should produce structured output with anatomical categories
        has_structure = any(
            kw in response.lower()
            for kw in ["lungs", "pleura", "cardiovascular", "airway", "other"]
        )
        if has_structure:
            print("PASSED: SRRG output contains anatomical categories")
        else:
            print("WARNING: SRRG output missing expected anatomical categories")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to CXR image")
    parser.add_argument("--srrg", action="store_true", help="Test SRRG variant")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image, srrg=args.srrg, device=args.device)

    # Test both variants if no --srrg flag
    if not args.srrg:
        print("\n" + "=" * 60 + "\n")
        validate(args.image, srrg=True, device=args.device)
