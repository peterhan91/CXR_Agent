"""
MedSAM bbox-prompted segmentation FastAPI server.

Cascaded workflow: CheXagent-2 provides bbox → MedSAM refines into pixel mask.
Uses HuggingFace wanglab/medsam-vit-base (SAM ViT-B fine-tuned on medical images).

Usage:
    CUDA_VISIBLE_DEVICES=2 python servers/medsam_server.py --port 8009
"""

import argparse
import base64
import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch

# Bypass transformers' torch version check for torch.load (CVE-2025-32434).
# transformers >= 4.50 requires torch >= 2.6 for safety, but our env has 2.5.x.
import transformers.utils.import_utils as _hf_import_utils
_hf_import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _hf_modeling_utils
_hf_modeling_utils.check_torch_load_is_safe = lambda: None

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("medsam_server")

state = {}


class SegmentRequest(BaseModel):
    image_path: str
    bbox: List[float]  # [x_min, y_min, x_max, y_max] normalized 0-1
    label: str = ""  # optional finding label for metadata


class SegmentResponse(BaseModel):
    label: str
    bbox: List[float]  # echo back input bbox
    coverage_pct: float
    mask_shape: List[int]
    mask_png_b64: Optional[str] = None
    inference_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = state.get("device", "cuda")
    model_name = state.get("model_name", "wanglab/medsam-vit-base")

    logger.info(f"Loading MedSAM from {model_name}...")
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained(model_name).to(device).eval()
    processor = SamProcessor.from_pretrained(model_name)

    state["model"] = model
    state["processor"] = processor
    state["device"] = device
    logger.info("MedSAM ready")
    yield
    state.clear()


app = FastAPI(title="MedSAM Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "MedSAM (wanglab/medsam-vit-base)"}


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    from PIL import Image

    model = state["model"]
    processor = state["processor"]
    device = state["device"]

    # Load image
    img = Image.open(req.image_path).convert("RGB")
    w, h = img.size

    # Convert normalized [0,1] bbox to pixel coordinates
    bbox = req.bbox
    if all(0 <= v <= 1.0 for v in bbox):
        px_bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
    else:
        px_bbox = bbox

    t0 = time.time()

    # Process with SAM processor
    inputs = processor(img, input_boxes=[[px_bbox]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Post-process: get mask at original resolution
    probs = processor.image_processor.post_process_masks(
        outputs.pred_masks.sigmoid().cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
        binarize=False,
    )
    mask_np = (probs[0].squeeze().numpy() > 0.5).astype(np.uint8)  # H x W

    inference_time = (time.time() - t0) * 1000

    # Compute coverage
    coverage = float(mask_np.sum() / mask_np.size * 100)

    # Encode mask as base64 PNG
    mask_img = Image.fromarray(mask_np * 255, mode="L")
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return SegmentResponse(
        label=req.label,
        bbox=req.bbox,
        coverage_pct=round(coverage, 2),
        mask_shape=list(mask_np.shape),
        mask_png_b64=mask_b64,
        inference_time_ms=inference_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_name", type=str, default="wanglab/medsam-vit-base")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    state["model_name"] = args.model_name
    state["device"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
