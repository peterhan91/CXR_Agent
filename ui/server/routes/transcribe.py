"""
/api/transcribe — proxy to the Whisper GPU server (port 8011).

The actual model runs in servers/whisper_server.py with torch.compile().
This route just forwards the audio upload and returns the transcribed text.
"""

import logging

import requests
from fastapi import APIRouter, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

WHISPER_URL = "http://localhost:8011/transcribe"


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Forward audio to Whisper server and return transcribed text."""
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(400, "Empty audio file")

        resp = requests.post(
            WHISPER_URL,
            files={"audio": (audio.filename or "recording.webm", content, audio.content_type or "audio/webm")},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"text": data["text"]}

    except requests.ConnectionError:
        raise HTTPException(503, "Whisper server not reachable. Is it running on :8011?")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription proxy failed: {e}", exc_info=True)
        raise HTTPException(500, f"Transcription failed: {e}")
