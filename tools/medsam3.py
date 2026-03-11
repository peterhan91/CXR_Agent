"""MedSAM3 text-guided segmentation tool."""

import base64
import hashlib
from pathlib import Path

import requests
from tools.base import BaseCXRTool

_MASK_DIR = Path("cache/masks/medsam3")
_MASK_DIR.mkdir(parents=True, exist_ok=True)


class MedSAM3SegmentTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8006"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medsam3_segment"

    @property
    def description(self) -> str:
        return (
            "Segment findings in a chest X-ray using MedSAM3 (text-guided SAM). "
            "Supports broader vocabulary than BiomedParse — can segment pleural effusion, "
            "pneumothorax, consolidation, endotracheal tube, central venous catheter, etc. "
            "Returns coverage percentage and mask shape for the segmented region."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the chest X-ray image file.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Clinical concept to segment (e.g., 'pleural effusion', 'endotracheal tube').",
                },
            },
            "required": ["image_path", "prompt"],
        }

    def run(self, image_path: str, prompt: str) -> str:
        resp = requests.post(
            f"{self.endpoint}/segment",
            json={"image_path": image_path, "prompt": prompt},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        lines = [
            f"MedSAM3 Segmentation for '{data['prompt']}':",
            f"  Coverage: {data['coverage_pct']:.1f}% of image",
            f"  Mask shape: {data['mask_shape']}",
        ]
        # Save mask to disk if available
        if data.get("mask_png_b64"):
            key = hashlib.md5(f"{image_path}_{prompt}".encode()).hexdigest()[:12]
            mask_path = _MASK_DIR / f"{key}.png"
            mask_path.write_bytes(base64.b64decode(data["mask_png_b64"]))
            lines.append(f"  Mask saved: {mask_path}")
        return "\n".join(lines)
