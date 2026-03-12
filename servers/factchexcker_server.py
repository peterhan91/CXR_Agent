"""
FactCheXcker report verification FastAPI server.

Two modes:
1. CarinaNet only — detects ETT/carina positions (no LLM needed)
2. Full pipeline — requires LLM backend for query/code generation

Usage:
    python servers/factchexcker_server.py --port 8007 \
        --factchexcker_dir ../FactCheXcker
"""

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("factchexcker_server")

state = {}


class CarinaRequest(BaseModel):
    image_path: str


class CarinaResponse(BaseModel):
    carina: Optional[dict] = None  # {x, y} or None if not detected
    ett: Optional[dict] = None
    inference_time_ms: float


class VerifyRequest(BaseModel):
    image_path: str
    report: str
    original_size: Optional[list] = None  # [H, W]; auto-detected from image if omitted
    pixel_spacing: list = [0.139, 0.139]


class VerifyResponse(BaseModel):
    updated_report: str
    changes_made: bool
    inference_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try importing CarinaNet
    try:
        import carinanet
        state["carinanet"] = True
        logger.info("CarinaNet available")
    except ImportError:
        state["carinanet"] = False
        logger.warning("CarinaNet not installed (pip install factchexcker-carinanet)")

    # Try importing full FactCheXcker pipeline
    factchexcker_dir = state.get("factchexcker_dir", "")
    if factchexcker_dir:
        sys.path.insert(0, factchexcker_dir)
        try:
            from api import CXRModuleRegistry, CXRImage
            config_path = Path(factchexcker_dir) / "configs" / "config.json"
            if config_path.exists():
                state["module_registry"] = CXRModuleRegistry(str(config_path))
                state["full_pipeline"] = True
                logger.info("Full FactCheXcker pipeline available")
            else:
                state["full_pipeline"] = False
                logger.warning(f"Config not found: {config_path}")
        except Exception as e:
            state["full_pipeline"] = False
            logger.warning(f"Full pipeline not available: {e}")

    logger.info("FactCheXcker server ready")
    yield
    state.clear()


app = FastAPI(title="FactCheXcker Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "carinanet": state.get("carinanet", False),
        "full_pipeline": state.get("full_pipeline", False),
    }


@app.post("/detect_carina", response_model=CarinaResponse)
async def detect_carina(req: CarinaRequest):
    if not state.get("carinanet"):
        return CarinaResponse(
            carina=None, ett=None, inference_time_ms=0,
        )

    import carinanet

    t0 = time.time()
    result = carinanet.predict_carina_ett(req.image_path)
    inference_time = (time.time() - t0) * 1000

    # Convert tuples (x, y) to dicts {x, y} for Pydantic
    carina_raw = result.get("carina")
    ett_raw = result.get("ett")
    carina = {"x": carina_raw[0], "y": carina_raw[1]} if isinstance(carina_raw, tuple) else carina_raw
    ett = {"x": ett_raw[0], "y": ett_raw[1]} if isinstance(ett_raw, tuple) else ett_raw

    return CarinaResponse(
        carina=carina,
        ett=ett,
        inference_time_ms=inference_time,
    )


@app.post("/verify_report", response_model=VerifyResponse)
async def verify_report(req: VerifyRequest):
    if not state.get("full_pipeline"):
        return VerifyResponse(
            updated_report=req.report,
            changes_made=False,
            inference_time_ms=0,
        )

    factchexcker_dir = state["factchexcker_dir"]
    sys.path.insert(0, factchexcker_dir)
    from api import CXRImage
    from FactCheXcker import FactCheXcker as Pipeline

    config_path = str(Path(factchexcker_dir) / "configs" / "config.json")

    # Auto-detect image dimensions if not provided
    original_size = req.original_size
    if original_size is None:
        from PIL import Image
        with Image.open(req.image_path) as img:
            w, h = img.size
            original_size = [h, w]

    cxr_image = CXRImage(
        rid="agent_verify",
        image_path=req.image_path,
        report=req.report,
        original_size=original_size,
        pixel_spacing=tuple(req.pixel_spacing),
        module_registry=state["module_registry"],
    )

    pipeline = Pipeline(config_path)

    t0 = time.time()
    updated_report = pipeline.run_pipeline(cxr_image)
    inference_time = (time.time() - t0) * 1000

    return VerifyResponse(
        updated_report=updated_report,
        changes_made=(updated_report != req.report),
        inference_time_ms=inference_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--factchexcker_dir", type=str, default="")
    args = parser.parse_args()

    state["factchexcker_dir"] = args.factchexcker_dir

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
