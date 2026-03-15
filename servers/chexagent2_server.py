"""
CheXagent-2 multi-task FastAPI server.

Serves report generation, structured findings, phrase grounding,
abnormality detection, classification, and VQA from a single process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python servers/chexagent2_server.py --port 8001
"""

import argparse
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


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

logger = logging.getLogger("chexagent2_server")

# Global model state
models = {}


# --- Task-specific prompt templates (from CheXagent GitHub) ---

PROMPTS = {
    "view_classification": (
        "What is the view of this chest X-ray? "
        "Options: (a) PA, (b) AP, (c) LATERAL"
    ),
    "binary_disease": "Does this chest X-ray contain {disease_name}?",
    "disease_identification": (
        "Given the CXR, identify any diseases. Options:\n{disease_names}"
    ),
    "phrase_grounding": "Please locate the following phrase: {phrase}",
    "abnormality_detection": (
        "Locate areas in the chest X-ray where {disease_name} are present, "
        "using box coordinates"
    ),
    "chest_tube_detection": (
        "Locate chest tubes and specify their positions with bounding box coordinates"
    ),
    "rib_fracture_detection": (
        "Locate rib fractures and specify their positions with bounding box coordinates"
    ),
    "foreign_objects_detection": (
        "Examine the chest X-ray for the presence of foreign objects, "
        "such as tubes, wires, pacemakers, or other devices. "
        "Specify their positions with bounding box coordinates"
    ),
    "srrg": "Structured Radiology Report Generation for Findings Section",
}


# --- Request / Response models ---

class ReportRequest(BaseModel):
    image_path: str
    prompt: str = "Generate a radiology report for this chest X-ray."
    max_new_tokens: int = 512


class SRRGRequest(BaseModel):
    image_path: str
    max_new_tokens: int = 512


class ClassifyRequest(BaseModel):
    image_path: str
    task: str = "view"  # "view", "binary_disease", "disease_id"
    disease_name: Optional[str] = None
    disease_names: Optional[list] = None
    max_new_tokens: int = 256


class GroundingRequest(BaseModel):
    image_path: str
    task: str = "phrase_grounding"  # "phrase_grounding", "abnormality", "chest_tube", "rib_fracture", "foreign_objects"
    phrase: Optional[str] = None
    disease_name: Optional[str] = None
    max_new_tokens: int = 512


class VQARequest(BaseModel):
    image_path: str
    question: str
    max_new_tokens: int = 512


class TextResponse(BaseModel):
    result: str
    generation_time_ms: float


class GroundingResponse(BaseModel):
    result: str
    boxes: list  # List of {x_min, y_min, x_max, y_max} normalized 0-1
    generation_time_ms: float


class ReportResponse(BaseModel):
    report: str
    generation_time_ms: float


# --- Helpers ---

def _parse_boxes(text: str) -> list:
    """Parse bounding box coordinates from CheXagent-2 output.

    CheXagent-2 outputs bounding boxes in two possible formats:
    1. <|box|> (x1,y1),(x2,y2) <|/box|>  — two 2-tuples (most common)
    2. (x1, y1, x2, y2) or [x1, y1, x2, y2]  — single 4-tuple

    Coordinates are on a 0-100 scale (percentage). Normalizes to 0-1.
    """
    boxes = []

    # Format 1: CheXagent-2 style <|box|> (x1,y1),(x2,y2) <|/box|>
    box_tags = re.findall(r'<\|box\|>\s*(.*?)\s*<\|/box\|>', text)
    for tag_content in box_tags:
        # Parse (x1,y1),(x2,y2) pairs within the tag
        pairs = re.findall(
            r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
            tag_content
        )
        # Each pair of 2-tuples forms one box
        for i in range(0, len(pairs) - 1, 2):
            x1, y1 = float(pairs[i][0]), float(pairs[i][1])
            x2, y2 = float(pairs[i+1][0]), float(pairs[i+1][1])
            scale = 1000.0 if any(c > 100 for c in [x1,y1,x2,y2]) else (100.0 if any(c > 1 for c in [x1,y1,x2,y2]) else 1.0)
            boxes.append({
                "x_min": x1 / scale,
                "y_min": y1 / scale,
                "x_max": x2 / scale,
                "y_max": y2 / scale,
            })

    if boxes:
        return boxes

    # Format 2: fallback — single 4-tuple (x1, y1, x2, y2)
    patterns = re.findall(
        r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]',
        text
    )
    for match in patterns:
        coords = [float(c) for c in match]
        scale = 1000.0 if any(c > 100 for c in coords) else (100.0 if any(c > 1 for c in coords) else 1.0)
        boxes.append({
            "x_min": coords[0] / scale,
            "y_min": coords[1] / scale,
            "x_max": coords[2] / scale,
            "y_max": coords[3] / scale,
        })
    return boxes


def _pad_to_square(image_path: str) -> tuple:
    """Pad an image to square with black borders (centered).

    CheXagent-2's visual encoder uses transforms.Resize((512, 512)) which
    distorts non-square images, causing bbox coordinates to be misaligned.
    Padding to square first preserves aspect ratio and produces correct coords.

    Returns (tmp_path, orig_w, orig_h, pad_left, pad_top).
    If already square, returns (image_path, w, h, 0, 0) with no temp file.
    """
    img = _load_cxr(image_path, mode="RGB")
    w, h = img.size

    if w == h:
        return image_path, w, h, 0, 0

    s = max(w, h)
    pad_left = (s - w) // 2
    pad_top = (s - h) // 2

    padded = Image.new("RGB", (s, s), (0, 0, 0))
    padded.paste(img, (pad_left, pad_top))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
    padded.save(tmp.name, format="PNG")
    tmp.close()

    return tmp.name, w, h, pad_left, pad_top


def _correct_boxes_for_padding(boxes: list, orig_w: int, orig_h: int,
                                pad_left: int, pad_top: int) -> list:
    """Map bbox coords from padded-square space back to original image space.

    The model outputs coords in [0,1] relative to the padded square.
    We subtract the padding offset and rescale to the original dimensions.
    """
    if pad_left == 0 and pad_top == 0:
        return boxes

    s = max(orig_w, orig_h)
    corrected = []
    for box in boxes:
        # Convert from padded-square [0,1] to pixel coords in padded image
        x_min_px = box["x_min"] * s
        y_min_px = box["y_min"] * s
        x_max_px = box["x_max"] * s
        y_max_px = box["y_max"] * s

        # Subtract padding offset to get pixel coords in original image
        x_min_orig = (x_min_px - pad_left) / orig_w
        y_min_orig = (y_min_px - pad_top) / orig_h
        x_max_orig = (x_max_px - pad_left) / orig_w
        y_max_orig = (y_max_px - pad_top) / orig_h

        # Clamp to [0, 1]
        x_min_orig = max(0.0, min(1.0, x_min_orig))
        y_min_orig = max(0.0, min(1.0, y_min_orig))
        x_max_orig = max(0.0, min(1.0, x_max_orig))
        y_max_orig = max(0.0, min(1.0, y_max_orig))

        # Skip degenerate boxes
        if x_max_orig <= x_min_orig or y_max_orig <= y_min_orig:
            continue

        corrected.append({
            "x_min": round(x_min_orig, 4),
            "y_min": round(y_min_orig, 4),
            "x_max": round(x_max_orig, 4),
            "y_max": round(y_max_orig, 4),
        })
    return corrected


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model = model.to(torch.bfloat16).eval()
    logger.info(f"{model_name} loaded")
    return model, tokenizer


def generate(model, tokenizer, image_paths: list, prompt: str,
             max_new_tokens: int = 512, device: str = "cuda") -> tuple:
    """Shared generation logic. Supports 0+ images."""
    parts = [{"image": p} for p in image_paths] + [{"text": prompt}]
    query = tokenizer.from_list_format(parts)
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    )

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids.to(device),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )[0]
    gen_time = (time.time() - t0) * 1000

    response = tokenizer.decode(output[input_ids.size(1):-1])
    return response, gen_time


# --- Lifespan ---

def _patch_visual_encoder(model):
    """Monkey-patch the visual encoder's load_image to handle 16-bit PNGs.

    CheXagent-2's VisualEncoder.load_image() uses PIL .convert("RGB") which
    silently clips 16-bit images. Replace with proper normalization.
    """
    import types
    import requests as _requests

    visual = model.transformer.visual if hasattr(model, "transformer") else None
    if visual is None:
        return

    original_transform = visual.image_transform

    def patched_load_image(self, image_path, training):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            img = Image.open(_requests.get(image_path, stream=True).raw)
        else:
            img = Image.open(image_path)
        # Properly handle 16-bit PNGs
        if img.mode in ("I", "I;16"):
            arr = np.array(img, dtype=np.float64)
            arr = arr - arr.min()
            mx = arr.max()
            if mx > 0:
                arr = (arr / mx * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = img.convert("RGB")
        return original_transform(img)

    visual.load_image = types.MethodType(patched_load_image, visual)
    logger.info("Patched VisualEncoder.load_image for 16-bit PNG support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["base"], models["base_tok"] = load_model("StanfordAIMI/CheXagent-2-3b")
    models["srrg"], models["srrg_tok"] = load_model(
        "StanfordAIMI/CheXagent-2-3b-srrg-findings"
    )
    # Patch both models for 16-bit PNG support
    _patch_visual_encoder(models["base"])
    _patch_visual_encoder(models["srrg"])
    yield
    models.clear()


app = FastAPI(title="CheXagent-2 Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "models": list(models.keys())}


# --- Report generation endpoints (existing) ---

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    response, gen_time = generate(
        models["base"], models["base_tok"],
        [req.image_path], req.prompt, req.max_new_tokens,
    )
    return ReportResponse(report=response, generation_time_ms=gen_time)


@app.post("/generate_srrg", response_model=ReportResponse)
async def generate_srrg(req: SRRGRequest):
    response, gen_time = generate(
        models["srrg"], models["srrg_tok"],
        [req.image_path], PROMPTS["srrg"], req.max_new_tokens,
    )
    return ReportResponse(report=response, generation_time_ms=gen_time)


# --- Classification endpoint (new) ---

@app.post("/classify", response_model=TextResponse)
async def classify(req: ClassifyRequest):
    if req.task == "view":
        prompt = PROMPTS["view_classification"]
    elif req.task == "binary_disease":
        prompt = PROMPTS["binary_disease"].format(disease_name=req.disease_name or "abnormality")
    elif req.task == "disease_id":
        names = ", ".join(req.disease_names or [])
        prompt = PROMPTS["disease_identification"].format(disease_names=names)
    else:
        prompt = f"Classify: {req.task}"

    response, gen_time = generate(
        models["base"], models["base_tok"],
        [req.image_path], prompt, req.max_new_tokens,
    )
    return TextResponse(result=response, generation_time_ms=gen_time)


# --- Grounding / detection endpoint (new) ---

@app.post("/ground", response_model=GroundingResponse)
async def ground(req: GroundingRequest):
    if req.task == "phrase_grounding":
        prompt = PROMPTS["phrase_grounding"].format(phrase=req.phrase or "abnormality")
    elif req.task == "abnormality":
        prompt = PROMPTS["abnormality_detection"].format(disease_name=req.disease_name or "abnormalities")
    elif req.task == "chest_tube":
        prompt = PROMPTS["chest_tube_detection"]
    elif req.task == "rib_fracture":
        prompt = PROMPTS["rib_fracture_detection"]
    elif req.task == "foreign_objects":
        prompt = PROMPTS["foreign_objects_detection"]
    else:
        prompt = f"Locate: {req.task}"

    # Pad image to square to prevent aspect ratio distortion in visual encoder
    tmp_path, orig_w, orig_h, pad_left, pad_top = _pad_to_square(req.image_path)
    try:
        response, gen_time = generate(
            models["base"], models["base_tok"],
            [tmp_path], prompt, req.max_new_tokens,
        )
    finally:
        if tmp_path != req.image_path:
            os.unlink(tmp_path)

    boxes = _parse_boxes(response)
    boxes = _correct_boxes_for_padding(boxes, orig_w, orig_h, pad_left, pad_top)
    return GroundingResponse(result=response, boxes=boxes, generation_time_ms=gen_time)


# --- VQA endpoint (new) ---

@app.post("/vqa", response_model=TextResponse)
async def vqa(req: VQARequest):
    response, gen_time = generate(
        models["base"], models["base_tok"],
        [req.image_path], req.question, req.max_new_tokens,
    )
    return TextResponse(result=response, generation_time_ms=gen_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
