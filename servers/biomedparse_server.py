"""
BiomedParse v1 FastAPI server for 2D CXR segmentation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005 \
        --biomedparse_dir ../BiomedParse
"""

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("biomedparse_server")

state = {}


class SegmentRequest(BaseModel):
    image_path: str
    prompts: list  # e.g., ["left lung", "lung opacity"]


class SegmentResult(BaseModel):
    prompt: str
    coverage_pct: float
    bbox: list  # [x_min, y_min, x_max, y_max] normalized 0-1
    mask_shape: list


class SegmentResponse(BaseModel):
    results: list  # List[SegmentResult]
    inference_time_ms: float


def mask_to_summary(mask: np.ndarray, prompt: str) -> SegmentResult:
    """Convert a binary mask to a text-friendly summary."""
    binary = (mask > 0.5).astype(np.float32)
    coverage = float(binary.sum() / binary.size * 100)

    # Bounding box (normalized)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = binary.shape
        bbox = [float(cmin / w), float(rmin / h), float(cmax / w), float(rmax / h)]
    else:
        bbox = [0, 0, 0, 0]

    return SegmentResult(
        prompt=prompt,
        coverage_pct=round(coverage, 2),
        bbox=bbox,
        mask_shape=list(binary.shape),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    biomedparse_dir = state.get("biomedparse_dir", "")
    sys.path.insert(0, biomedparse_dir)

    logger.info("Loading BiomedParse v1...")
    from modeling.BaseModel import BaseModel as BPBaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files

    import os
    config_path = os.path.join(biomedparse_dir, "configs", "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    model = BPBaseModel(opt, build_model(opt)).from_pretrained(
        "hf_hub:microsoft/BiomedParse"
    ).eval().cuda()

    state["model"] = model
    logger.info("BiomedParse ready")
    yield
    state.clear()


app = FastAPI(title="BiomedParse Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "BiomedParse-v1"}


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    from PIL import Image
    from inference_utils.inference import interactive_infer_image

    model = state["model"]
    image = Image.open(req.image_path).convert("RGB")

    t0 = time.time()
    pred_masks = interactive_infer_image(model, image, req.prompts)
    inference_time = (time.time() - t0) * 1000

    results = []
    for prompt, mask in zip(req.prompts, pred_masks):
        results.append(mask_to_summary(mask, prompt))

    return SegmentResponse(results=results, inference_time_ms=inference_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--biomedparse_dir", type=str, required=True)
    args = parser.parse_args()

    state["biomedparse_dir"] = args.biomedparse_dir

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
