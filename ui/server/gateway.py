"""
CXR Agent API Gateway.

Thin FastAPI layer that serves existing eval results, trajectories, images,
and (later) wraps the live agent for RITL experiments.

Usage:
    conda run -n cxr_agent python ui/server/gateway.py --port 9000
"""

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from routes.results import router as results_router
from routes.run import router as run_router
from routes.transcribe import router as transcribe_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "eval_v4"

ALLOWED_IMAGE_ROOTS = [
    "/home/than/physionet.org",
    "/home/than/Datasets",
    "/home/than/.cache",
    "/home/than/DeepLearning/CXR_Agent/data",
    "/home/than/DeepLearning/CXR_Agent/results",
]

app = FastAPI(title="CXR Agent Gateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(results_router, prefix="/api")
app.include_router(run_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")


@app.get("/api/image")
async def serve_image(path: str):
    """Serve a CXR image file. Path must be under allowed roots."""
    resolved = os.path.realpath(path)
    if not any(resolved.startswith(root) for root in ALLOWED_IMAGE_ROOTS):
        raise HTTPException(403, "Access denied: path outside allowed directories")
    if not resolved.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Only PNG/JPEG files allowed")
    if not os.path.isfile(resolved):
        raise HTTPException(404, f"Image not found: {path}")
    return FileResponse(resolved, media_type="image/jpeg")


@app.get("/api/health")
async def health():
    return {"status": "ok", "results_dir": str(RESULTS_DIR)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
