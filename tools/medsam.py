"""MedSAM bbox-prompted segmentation tool."""

import base64
import hashlib
from pathlib import Path

import requests
from tools.base import BaseCXRTool

_MASK_DIR = Path("cache/masks/medsam")
_MASK_DIR.mkdir(parents=True, exist_ok=True)


class MedSAMSegmentTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8009"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medsam_segment"

    @property
    def description(self) -> str:
        return (
            "Segment a finding in a chest X-ray using MedSAM (SAM fine-tuned on medical images). "
            "REQUIRES a bounding box as input — use chexagent2_grounding first to get the bbox, "
            "then pass it here for precise pixel-level segmentation within that region. "
            "Cascaded workflow: chexagent2_grounding → medsam_segment. "
            "Returns coverage percentage, mask shape, and saves the mask to disk."
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
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Bounding box [x_min, y_min, x_max, y_max] in normalized [0,1] coordinates. "
                        "Get this from chexagent2_grounding first."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Finding label (e.g., 'cardiomegaly', 'pleural effusion') for metadata.",
                    "default": "",
                },
            },
            "required": ["image_path", "bbox"],
        }

    def run(self, image_path: str, bbox: list, label: str = "") -> str:
        resp = requests.post(
            f"{self.endpoint}/segment",
            json={"image_path": image_path, "bbox": bbox, "label": label},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        lines = [
            f"MedSAM Segmentation for '{data['label'] or 'region'}':",
            f"  Input bbox: [{', '.join(f'{v:.2f}' for v in data['bbox'])}]",
            f"  Coverage: {data['coverage_pct']:.1f}% of image",
            f"  Mask shape: {data['mask_shape']}",
        ]

        # Save mask to disk if available
        if data.get("mask_png_b64"):
            key = hashlib.md5(f"{image_path}_{bbox}_{label}".encode()).hexdigest()[:12]
            mask_path = _MASK_DIR / f"{key}.png"
            mask_path.write_bytes(base64.b64decode(data["mask_png_b64"]))
            lines.append(f"  Mask saved: {mask_path}")

        return "\n".join(lines)
