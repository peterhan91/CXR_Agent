"""
Validate CheXOne (Qwen2.5-VL-3B based) inference.

Run on GPU server:
    python scripts/validate_models/validate_chexone.py --image path/to/cxr.jpg

Requires:
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install qwen-vl-utils
"""

import argparse
import tempfile
import time
import torch


def download_sample_image():
    """Download a sample CXR image for testing."""
    import requests
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        return f.name


def validate_reasoning(image_path: str, model, processor):
    """Test reasoning mode (step-by-step with \\boxed{})."""
    print("\n=== Reasoning Mode ===")
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Write an example findings section for the CXR. "
                            "Please reason step by step, and put your final answer within \\boxed{}.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    gen_time = time.time() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_text[0]

    print(f"Generation time: {gen_time:.1f}s")
    print(f"Output tokens: {len(generated_ids_trimmed[0])}")
    print(f"\n--- Response ---\n{response}\n--- End ---\n")

    assert isinstance(response, str) and len(response) > 20, "Response too short"
    print("PASSED: Response is a non-trivial string")

    if "\\boxed" in response or "boxed" in response:
        print("PASSED: Reasoning mode produced boxed answer")
    else:
        print("WARNING: No \\boxed{} in reasoning output (may still be valid)")

    return response


def validate_instruct(image_path: str, model, processor):
    """Test instruct mode (direct answer, no reasoning trace)."""
    print("\n=== Instruct Mode ===")
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Write an example findings section for the CXR.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    gen_time = time.time() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_text[0]

    print(f"Generation time: {gen_time:.1f}s")
    print(f"\n--- Response ---\n{response}\n--- End ---\n")

    assert isinstance(response, str) and len(response) > 20, "Response too short"
    print("PASSED: Response is a non-trivial string")

    return response


def validate(image_path: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_name = "StanfordAIMI/CheXOne"
    print(f"=== Validating {model_name} ===")

    print("Loading model...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)

    # Test both modes
    validate_reasoning(image_path, model, processor)
    validate_instruct(image_path, model, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to CXR image")
    args = parser.parse_args()

    if args.image is None:
        print("No image provided, downloading sample...")
        args.image = download_sample_image()
        print(f"Sample image: {args.image}")

    validate(args.image)
