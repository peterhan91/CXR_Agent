"""
MedSAM3 text-guided segmentation FastAPI server.

MedSAM3 may only support CLI inference (infer_sam.py). This server wraps
the CLI as a subprocess if no Python API is available.

Usage:
    CUDA_VISIBLE_DEVICES=2 python servers/medsam3_server.py --port 8006 \
        --medsam3_dir ../MedSAM3
"""

import argparse
import base64
import json
import logging
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("medsam3_server")

state = {}


class SegmentRequest(BaseModel):
    image_path: str
    prompt: str  # e.g., "pleural effusion"
    threshold: float = 0.5


class SegmentResponse(BaseModel):
    prompt: str
    coverage_pct: float
    mask_shape: list
    mask_png_b64: Optional[str] = None  # base64-encoded PNG mask
    inference_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    medsam3_dir = state.get("medsam3_dir", "")
    medsam3_path = Path(medsam3_dir)

    # Verify the repo exists
    infer_script = medsam3_path / "infer_sam.py"
    config_file = medsam3_path / "configs" / "full_lora_config.yaml"

    if not infer_script.exists():
        logger.error(f"infer_sam.py not found at {infer_script}")
        raise FileNotFoundError(f"MedSAM3 not found at {medsam3_dir}")

    state["infer_script"] = str(infer_script)
    state["config_file"] = str(config_file) if config_file.exists() else None

    # Try to import MedSAM3 as Python module
    sys.path.insert(0, medsam3_dir)
    try:
        from medsam3 import MedSAM3Model
        state["python_api"] = True
        logger.info("MedSAM3 Python API available")
    except ImportError:
        state["python_api"] = False
        logger.info("MedSAM3 Python API not available, using CLI wrapper")

    logger.info("MedSAM3 server ready")
    yield
    state.clear()


app = FastAPI(title="MedSAM3 Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "MedSAM3",
        "mode": "python" if state.get("python_api") else "cli",
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    """Segment via CLI subprocess."""
    output_path = tempfile.mktemp(suffix=".png")

    cmd = [
        sys.executable, state["infer_script"],
        "--image", req.image_path,
        "--prompt", req.prompt,
        "--threshold", str(req.threshold),
        "--output", output_path,
    ]
    if state.get("config_file"):
        cmd.extend(["--config", state["config_file"]])

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
        cwd=state.get("medsam3_dir", None),
    )
    inference_time = (time.time() - t0) * 1000

    if result.returncode != 0:
        logger.error(f"MedSAM3 failed: {result.stderr[:500]}")
        return SegmentResponse(
            prompt=req.prompt, coverage_pct=0.0,
            mask_shape=[0, 0], inference_time_ms=inference_time,
        )

    # Read output mask
    from PIL import Image
    output_file = Path(output_path)
    mask_b64 = None
    if output_file.exists():
        mask = np.array(Image.open(output_path).convert("L"))
        binary = (mask > 127).astype(np.float32)
        coverage = float(binary.sum() / binary.size * 100)
        mask_shape = list(binary.shape)
        # Capture mask as base64 PNG before cleanup
        mask_b64 = base64.b64encode(output_file.read_bytes()).decode("utf-8")
        output_file.unlink()  # cleanup
    else:
        coverage = 0.0
        mask_shape = [0, 0]

    return SegmentResponse(
        prompt=req.prompt,
        coverage_pct=round(coverage, 2),
        mask_shape=mask_shape,
        mask_png_b64=mask_b64,
        inference_time_ms=inference_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--medsam3_dir", type=str, required=True)
    args = parser.parse_args()

    state["medsam3_dir"] = args.medsam3_dir

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
