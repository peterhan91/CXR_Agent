"""
CheXagent-2 multi-task FastAPI server.

Serves report generation, structured findings, phrase grounding,
abnormality detection, classification, and VQA from a single process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python servers/chexagent2_server.py --port 8001
"""

import argparse
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

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
    """Parse bounding box coordinates from model output.

    CheXagent-2 typically outputs coordinates as (x1, y1, x2, y2) on a
    0-1000 or 0-100 scale. This parser handles both and normalizes to 0-1.
    """
    boxes = []
    # Match patterns like (123, 456, 789, 012) or [123, 456, 789, 012]
    patterns = re.findall(
        r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]',
        text
    )
    for match in patterns:
        coords = [float(c) for c in match]
        # Determine scale: if any coord > 1, normalize
        scale = 1000.0 if any(c > 100 for c in coords) else (100.0 if any(c > 1 for c in coords) else 1.0)
        boxes.append({
            "x_min": coords[0] / scale,
            "y_min": coords[1] / scale,
            "x_max": coords[2] / scale,
            "y_max": coords[3] / scale,
        })
    return boxes


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["base"], models["base_tok"] = load_model("StanfordAIMI/CheXagent-2-3b")
    models["srrg"], models["srrg_tok"] = load_model(
        "StanfordAIMI/CheXagent-2-3b-srrg-findings"
    )
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

    response, gen_time = generate(
        models["base"], models["base_tok"],
        [req.image_path], prompt, req.max_new_tokens,
    )
    boxes = _parse_boxes(response)
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
