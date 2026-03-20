"""
Whisper large-v3-turbo FastAPI server with torch.compile().

Model: openai/whisper-large-v3-turbo (809M params, MIT license)
Uses static cache + torch.compile for ~4.5x inference speedup.

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/whisper_server.py --port 8011
"""

import argparse
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("whisper_server")

state = {}

MODEL_ID = "openai/whisper-large-v3-turbo"


class TranscribeResponse(BaseModel):
    text: str
    language: str
    generation_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading {MODEL_ID} on {device}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # Enable static cache + torch.compile for speed
    if device.startswith("cuda"):
        logger.info("Applying torch.compile (reduce-overhead)...")
        torch.set_float32_matmul_precision("high")
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 448
        model.forward = torch.compile(
            model.forward, mode="reduce-overhead", fullgraph=True
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Warm-up passes to trigger compilation
    if device.startswith("cuda"):
        logger.info("Running warm-up passes for torch.compile...")
        import numpy as np
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Generate a short silent audio for warm-up
        dummy_audio = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
        for i in range(2):
            with sdpa_kernel(SDPBackend.MATH):
                pipe(
                    dummy_audio.copy(),
                    generate_kwargs={"min_new_tokens": 4, "max_new_tokens": 4},
                )
            logger.info(f"  Warm-up {i + 1}/2 done")

    state["pipe"] = pipe
    state["device"] = device
    logger.info("Whisper ready")
    yield
    state.clear()


app = FastAPI(title="Whisper Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": state.get("device", "unknown")}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe uploaded audio file to text."""
    pipe = state.get("pipe")
    if not pipe:
        raise HTTPException(503, "Model not loaded")

    try:
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            if not content:
                raise HTTPException(400, "Empty audio file")
            tmp.write(content)
            tmp_path = tmp.name

        t0 = time.time()

        # Use MATH SDP backend for torch.compile compatibility
        if state["device"].startswith("cuda"):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                result = pipe(tmp_path, generate_kwargs={"language": "english"})
        else:
            result = pipe(tmp_path, generate_kwargs={"language": "english"})

        gen_time = (time.time() - t0) * 1000

        Path(tmp_path).unlink(missing_ok=True)

        return TranscribeResponse(
            text=result["text"].strip(),
            language="english",
            generation_time_ms=gen_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(500, f"Transcription failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
