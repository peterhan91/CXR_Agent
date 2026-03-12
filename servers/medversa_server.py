"""
MedVersa (7B) multi-task FastAPI server.

Supports report generation, VQA, classification, detection, and 2D segmentation.

Requires the MedVersa repo cloned with its environment:
    git clone https://huggingface.co/hyzhou/MedVersa
    cd MedVersa && conda env create -f environment.yml

Usage:
    CUDA_VISIBLE_DEVICES=2 python servers/medversa_server.py --port 8004 \
        --medversa_dir ../MedVersa
"""

import argparse
import base64
import logging
import sys
import time
import types
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

# ---------------------------------------------------------------------------
# Monkey-patch ipdb/pdb to prevent set_trace() calls in MedVersa from
# triggering bdb.BdbQuit and crashing the server.  MedVersa ships with
# ipdb.set_trace() left in its inference path (medomni/models/medomni.py:292
# among others).  We install a fake ipdb module so every `import ipdb` and
# `ipdb.set_trace()` becomes a silent no-op.
# ---------------------------------------------------------------------------
_fake_ipdb = types.ModuleType("ipdb")
_fake_ipdb.set_trace = lambda *a, **kw: None
_fake_ipdb.launch_ipdb_on_exception = lambda *a, **kw: None
_fake_ipdb.post_mortem = lambda *a, **kw: None
_fake_ipdb.pm = lambda *a, **kw: None
_fake_ipdb.run = lambda *a, **kw: None
_fake_ipdb.runcall = lambda *a, **kw: None
_fake_ipdb.runeval = lambda *a, **kw: None
sys.modules["ipdb"] = _fake_ipdb

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("medversa_server")

state = {}

# Default generation parameters
GEN_PARAMS = {
    "num_beams": 1,
    "do_sample": True,
    "min_length": 1,
    "top_p": 0.9,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "temperature": 0.1,
}


# --- Request / Response models ---

class ReportRequest(BaseModel):
    image_path: str
    context: str = ""
    prompt: str = "How would you characterize the findings from <img0>?"
    modality: str = "cxr"
    max_new_tokens: int = 512


class VQARequest(BaseModel):
    image_path: str
    question: str
    context: str = ""
    modality: str = "cxr"


class ClassifyRequest(BaseModel):
    image_path: str
    context: str = ""
    modality: str = "cxr"


class DetectRequest(BaseModel):
    image_path: str
    prompt: str = "Can you detect any abnormality on <img0>? Please output the bounding box."
    context: str = ""
    modality: str = "cxr"


class SegmentRequest(BaseModel):
    image_path: str
    prompt: str = "Can you segment the abnormal region in <img0>?"
    context: str = ""
    modality: str = "cxr"


class ReportResponse(BaseModel):
    report: str
    generation_time_ms: float


class TextResponse(BaseModel):
    result: str
    generation_time_ms: float


class DetectResponse(BaseModel):
    result: str
    generation_time_ms: float


class SegmentResponse(BaseModel):
    result: str
    has_mask: bool
    coverage_pct: float
    mask_shape: list
    mask_png_b64: Optional[str] = None  # base64-encoded PNG mask
    generation_time_ms: float


# --- Helpers ---

def _mask_summary(mask) -> dict:
    """Convert segmentation mask to coverage stats."""
    if mask is None:
        return {"has_mask": False, "coverage_pct": 0.0, "mask_shape": [0, 0]}
    arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
    if arr.size == 0:
        return {"has_mask": False, "coverage_pct": 0.0, "mask_shape": [0, 0]}
    binary = (arr > 0.5).astype(np.float32)
    coverage = float(binary.sum() / binary.size * 100)
    return {
        "has_mask": True,
        "coverage_pct": round(coverage, 2),
        "mask_shape": list(binary.shape[:2]),
    }


def _generate(image_path: str, context: str, prompt: str,
              modality: str, task: str) -> tuple:
    """Shared generation logic for all MedVersa tasks."""
    model = state["model"]
    generate_fn = state["generate_predictions"]
    device = state.get("device", "cuda")

    t0 = time.time()
    seg_mask_2d, seg_mask_3d, output_text = generate_fn(
        model,
        [image_path],
        context,
        prompt,
        modality,
        task,
        device=device,
        **GEN_PARAMS,
    )
    gen_time = (time.time() - t0) * 1000
    return seg_mask_2d, output_text, gen_time


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    medversa_dir = state.get("medversa_dir", "")
    device = state.get("device", "cuda")

    sys.path.insert(0, medversa_dir)

    logger.info("Loading MedVersa model...")
    from utils import registry, generate_predictions

    model_cls = registry.get_model_class("medomni")
    import os
    from huggingface_hub import HfFolder
    token = os.environ.get("HF_TOKEN", HfFolder.get_token())
    model = model_cls.from_pretrained("hyzhou/MedVersa_Internal", token=token).to(device).eval()

    state["model"] = model
    state["generate_predictions"] = generate_predictions
    logger.info("MedVersa loaded")
    yield
    state.clear()


app = FastAPI(title="MedVersa Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "MedVersa"}


# --- Report generation (existing) ---

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    _, output_text, gen_time = _generate(
        req.image_path, req.context, req.prompt, req.modality, "report",
    )
    return ReportResponse(report=output_text, generation_time_ms=gen_time)


# --- VQA (new) ---

@app.post("/vqa", response_model=TextResponse)
async def vqa(req: VQARequest):
    prompt = f"{req.question} <img0>"
    _, output_text, gen_time = _generate(
        req.image_path, req.context, prompt, req.modality, "vqa",
    )
    return TextResponse(result=output_text, generation_time_ms=gen_time)


# --- Classification (new) ---

@app.post("/classify", response_model=TextResponse)
async def classify(req: ClassifyRequest):
    prompt = "What are the possible diagnoses of <img0>?"
    _, output_text, gen_time = _generate(
        req.image_path, req.context, prompt, req.modality, "classification",
    )
    return TextResponse(result=output_text, generation_time_ms=gen_time)


# --- Detection (new) ---

@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    _, output_text, gen_time = _generate(
        req.image_path, req.context, req.prompt, req.modality, "detection",
    )
    return DetectResponse(result=output_text, generation_time_ms=gen_time)


# --- 2D Segmentation (new) ---

@app.post("/segment_2d", response_model=SegmentResponse)
async def segment_2d(req: SegmentRequest):
    seg_mask_2d, output_text, gen_time = _generate(
        req.image_path, req.context, req.prompt, req.modality, "2d segmentation",
    )
    summary = _mask_summary(seg_mask_2d)

    # Encode mask as base64 PNG
    mask_b64 = None
    if summary["has_mask"] and seg_mask_2d is not None:
        from PIL import Image as PILImage
        arr = np.array(seg_mask_2d) if not isinstance(seg_mask_2d, np.ndarray) else seg_mask_2d
        if arr.size > 0:
            # Normalize to 0-255
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                normalized = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                normalized = (arr * 255).astype(np.uint8)
            pil_mask = PILImage.fromarray(normalized, mode="L")
            buf = BytesIO()
            pil_mask.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return SegmentResponse(
        result=output_text,
        has_mask=summary["has_mask"],
        coverage_pct=summary["coverage_pct"],
        mask_shape=summary["mask_shape"],
        mask_png_b64=mask_b64,
        generation_time_ms=gen_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--medversa_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    state["medversa_dir"] = args.medversa_dir
    state["device"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
