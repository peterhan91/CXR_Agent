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

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

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

    image = Image.open(image_path).convert("RGB")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
