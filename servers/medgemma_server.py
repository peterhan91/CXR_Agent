"""
MedGemma (google/medgemma-1.5-4b-it) FastAPI server.

Supports VQA, report generation, and classification on CXR images.

Usage:
    CUDA_VISIBLE_DEVICES=2 python servers/medgemma_server.py --port 8010
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from PIL import Image


def _load_cxr(image_path: str, mode: str = "RGB") -> Image.Image:
    """Load CXR image, properly normalizing 16-bit PNGs to 8-bit.

    PIL's .convert() silently clips 16-bit (mode I) images to 8-bit,
    destroying the dynamic range. PadChest-GR and RexGradient use 16-bit PNGs.
    """
    img = Image.open(image_path)
    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float64)
        arr = arr - arr.min()
        mx = arr.max()
        if mx > 0:
            arr = (arr / mx * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        img = img.convert("L")
    if mode == "RGB":
        img = img.convert("RGB")
    return img

logger = logging.getLogger("medgemma_server")

state = {}


class VQARequest(BaseModel):
    image_path: str
    question: str
    max_new_tokens: int = 1024


class VQAResponse(BaseModel):
    answer: str
    generation_time_ms: float


class ReportRequest(BaseModel):
    image_path: str
    prompt: str = "Describe the findings in this chest X-ray."
    max_new_tokens: int = 1024


class ReportResponse(BaseModel):
    report: str
    generation_time_ms: float


class GroundingRequest(BaseModel):
    image_path: str
    phrase: str
    max_new_tokens: int = 256


class GroundingResponse(BaseModel):
    result: str
    boxes: list  # list of {x_min, y_min, x_max, y_max} normalized 0-1
    generation_time_ms: float


class LongitudinalRequest(BaseModel):
    current_image_path: str
    prior_image_path: str
    max_new_tokens: int = 1024


class LongitudinalResponse(BaseModel):
    comparison: str
    generation_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "google/medgemma-1.5-4b-it"
    logger.info(f"Loading {model_id}...")
    state["model"] = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    state["processor"] = AutoProcessor.from_pretrained(model_id)
    logger.info("MedGemma loaded")
    yield
    state.clear()


app = FastAPI(title="MedGemma Server", lifespan=lifespan)


def _generate(image_path: str, prompt: str, max_new_tokens: int) -> tuple:
    """Shared generation logic for VQA and report endpoints."""
    from PIL import Image

    model = state["model"]
    processor = state["processor"]

    image = _load_cxr(image_path, mode="RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_time = (time.time() - t0) * 1000

    # Trim input tokens from output
    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][input_len:]
    text = processor.decode(output_ids, skip_special_tokens=True)

    return text.strip(), gen_time


@app.get("/health")
async def health():
    return {"status": "ok", "model": "MedGemma-1.5-4B-IT"}


@app.post("/vqa", response_model=VQAResponse)
async def vqa(req: VQARequest):
    answer, gen_time = _generate(req.image_path, req.question, req.max_new_tokens)
    return VQAResponse(answer=answer, generation_time_ms=gen_time)


@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    report, gen_time = _generate(req.image_path, req.prompt, req.max_new_tokens)
    return ReportResponse(report=report, generation_time_ms=gen_time)


def _generate_multi_image(image_paths: list, prompt: str, max_new_tokens: int) -> tuple:
    """Generation with multiple images."""
    model = state["model"]
    processor = state["processor"]

    content = []
    for p in image_paths:
        img = _load_cxr(p, mode="RGB")
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_time = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][input_len:]
    text = processor.decode(output_ids, skip_special_tokens=True)
    return text.strip(), gen_time


def _pad_to_square(image_path: str) -> tuple:
    """Pad image to square, return (padded_img, orig_w, orig_h, pad_left, pad_top)."""
    img = _load_cxr(image_path, mode="RGB")
    w, h = img.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    pad_left = (size - w) // 2
    pad_top = (size - h) // 2
    padded.paste(img, (pad_left, pad_top))
    return padded, w, h, pad_left, pad_top


def _parse_grounding_boxes(text: str, orig_w: int, orig_h: int, pad_left: int, pad_top: int) -> list:
    """Parse MedGemma grounding output and correct for square padding.

    MedGemma outputs bounding boxes as [y0, x0, y1, x1] in [0, 1000] range.
    We convert to {x_min, y_min, x_max, y_max} normalized to [0, 1] in original image coords.
    """
    import re
    boxes = []
    # Match patterns like [123, 456, 789, 012] or <box>123 456 789 012</box>
    for match in re.finditer(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
        y0, x0, y1, x1 = [int(v) for v in match.groups()]
        # Convert from [0, 1000] to [0, 1] in padded square
        size = max(orig_w, orig_h)
        x_min_pad = x0 / 1000.0
        y_min_pad = y0 / 1000.0
        x_max_pad = x1 / 1000.0
        y_max_pad = y1 / 1000.0

        # Correct for padding: convert from padded square to original image
        x_min = (x_min_pad * size - pad_left) / orig_w
        y_min = (y_min_pad * size - pad_top) / orig_h
        x_max = (x_max_pad * size - pad_left) / orig_w
        y_max = (y_max_pad * size - pad_top) / orig_h

        # Clip to [0, 1]
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        if x_max > x_min and y_max > y_min:
            boxes.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    return boxes


@app.post("/ground", response_model=GroundingResponse)
async def ground(req: GroundingRequest):
    """Ground a phrase in a chest X-ray, returning bounding boxes."""
    padded_img, orig_w, orig_h, pad_left, pad_top = _pad_to_square(req.image_path)

    model = state["model"]
    processor = state["processor"]

    prompt = (
        f"Where is {req.phrase} in this chest X-ray? "
        f"Output bounding box coordinates as [y0, x0, y1, x1] in the range [0, 1000]. "
        f"If {req.phrase} is not present, say 'Not detected.'"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": padded_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    gen_time = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][input_len:]
    result = processor.decode(output_ids, skip_special_tokens=True).strip()

    boxes = _parse_grounding_boxes(result, orig_w, orig_h, pad_left, pad_top)
    return GroundingResponse(result=result, boxes=boxes, generation_time_ms=gen_time)


@app.post("/longitudinal", response_model=LongitudinalResponse)
async def longitudinal(req: LongitudinalRequest):
    """Compare current and prior chest X-rays for interval changes."""
    prompt = (
        "Compare these two chest X-rays. The first is the prior study and the second "
        "is the current study. Describe interval changes including any findings that "
        "have improved, worsened, or remained stable."
    )
    comparison, gen_time = _generate_multi_image(
        [req.prior_image_path, req.current_image_path],
        prompt,
        req.max_new_tokens,
    )
    return LongitudinalResponse(comparison=comparison, generation_time_ms=gen_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
