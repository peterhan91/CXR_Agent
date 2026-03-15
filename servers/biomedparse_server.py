"""
BiomedParse v1 FastAPI server for 2D CXR segmentation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005 \
        --biomedparse_dir ../BiomedParse
"""

import argparse
import base64
import logging
import sys
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import torch

# Bypass transformers' torch version check for torch.load (CVE-2025-32434).
# transformers >= 4.50 requires torch >= 2.6, but BiomedParse v1 uses legacy checkpoints.
# Patch both locations where the check is referenced.
import transformers.utils.import_utils as _hf_import_utils
_hf_import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _hf_modeling_utils
_hf_modeling_utils.check_torch_load_is_safe = lambda: None

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image as PILImage


def _load_cxr(image_path: str, mode: str = "RGB") -> "PILImage.Image":
    """Load CXR image, properly normalizing 16-bit PNGs to 8-bit.

    PIL's .convert() silently clips 16-bit (mode I) images to 8-bit,
    destroying the dynamic range. PadChest-GR and RexGradient use 16-bit PNGs.
    """
    img = PILImage.open(image_path)
    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float64)
        arr = arr - arr.min()
        mx = arr.max()
        if mx > 0:
            arr = (arr / mx * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = PILImage.fromarray(arr, mode="L")
    else:
        img = img.convert("L")
    if mode == "RGB":
        img = img.convert("RGB")
    return img

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
    mask_png_b64: Optional[str] = None  # base64-encoded PNG mask


class SegmentResponse(BaseModel):
    results: list  # List[SegmentResult]
    inference_time_ms: float


def mask_to_summary(mask: np.ndarray, prompt: str) -> SegmentResult:
    """Convert a binary mask to a text-friendly summary + base64 PNG."""
    from PIL import Image as PILImage

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

    # Encode binary mask as PNG (base64)
    pil_mask = PILImage.fromarray((binary * 255).astype(np.uint8), mode="L")
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return SegmentResult(
        prompt=prompt,
        coverage_pct=round(coverage, 2),
        bbox=bbox,
        mask_shape=list(binary.shape),
        mask_png_b64=mask_b64,
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
    orig_dir = os.getcwd()
    os.chdir(biomedparse_dir)

    config_path = os.path.join(biomedparse_dir, "configs", "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    model = BPBaseModel(opt, build_model(opt)).from_pretrained(
        "hf_hub:microsoft/BiomedParse"
    ).eval().cuda()

    # Pre-compute default text embeddings for BIOMED_CLASSES so that
    # compute_similarity() can find self.default_text_embeddings during inference.
    # Without this, evaluate_demo -> compute_similarity raises:
    #   AttributeError: 'LanguageEncoder' object has no attribute 'default_text_embeddings'
    from utilities.constants import BIOMED_CLASSES
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    logger.info("Default text embeddings computed for %d classes", len(BIOMED_CLASSES) + 1)

    # Import inference_utils while cwd is still in biomedparse_dir,
    # because output_processing.py loads target_dist.json via relative path.
    from inference_utils.inference import interactive_infer_image
    state["interactive_infer_image"] = interactive_infer_image

    os.chdir(orig_dir)

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

    model = state["model"]
    interactive_infer_image = state["interactive_infer_image"]
    image = _load_cxr(req.image_path, mode="RGB")

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
