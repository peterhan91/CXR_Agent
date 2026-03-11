"""
CheXOne (Qwen2.5-VL-3B) FastAPI server.

Supports both reasoning mode (step-by-step + boxed answer) and
instruct mode (direct report).

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("chexone_server")

state = {}


REASONING_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


class ReportRequest(BaseModel):
    image_path: str
    prompt: str = "Write the findings section for this chest X-ray."
    reasoning: bool = False
    max_new_tokens: int = 1024


class ReportResponse(BaseModel):
    report: str
    generation_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_name = "StanfordAIMI/CheXOne"
    logger.info(f"Loading {model_name}...")
    state["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    ).eval()
    state["processor"] = AutoProcessor.from_pretrained(model_name)
    logger.info("CheXOne loaded")
    yield
    state.clear()


app = FastAPI(title="CheXOne Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "CheXOne"}


@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    from qwen_vl_utils import process_vision_info

    model = state["model"]
    processor = state["processor"]

    prompt_text = req.prompt
    if req.reasoning:
        prompt_text += REASONING_SUFFIX

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": req.image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    gen_time = (time.time() - t0) * 1000

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return ReportResponse(report=output_text[0], generation_time_ms=gen_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
