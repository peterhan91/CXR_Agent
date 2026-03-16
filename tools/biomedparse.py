"""BiomedParse segmentation tool."""

import base64
import hashlib
from pathlib import Path

import requests
from tools.base import BaseCXRTool

_MASK_DIR = Path("cache/masks/biomedparse")
_MASK_DIR.mkdir(parents=True, exist_ok=True)


class BiomedParseSegmentTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8005"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "biomedparse_segment"

    @property
    def description(self) -> str:
        return (
            "[GROUNDING] "
            "Segment structures/findings in a chest X-ray using BiomedParse (Microsoft, FocalNet-Large). "
            "Returns coverage percentage, bounding box [0-1], and mask PNG per prompt. ~0.4s inference. "
            "Verified CXR prompts: 'left lung', 'right lung', 'lung', 'lung opacity', "
            "'viral pneumonia', 'COVID-19 infection'. Other prompts may work but are UNTESTED. "
            "COVID-biased training — non-COVID pathology segmentation may be unreliable. "
            "WHEN TO USE: Use for diffuse/bilateral findings where you need extent quantification "
            "(e.g., 'how much of the lung is opacified?'). For focal findings, prefer chexagent2_grounding. "
            "EXAMPLE: "
            "Input: {prompts: ['left lung', 'right lung', 'lung opacity']} → "
            "'BiomedParse Segmentation Results:\n"
            "  'left lung': 9.6% coverage, bbox=[0.61, 0.12, 0.94, 0.67]\n"
            "    Mask saved: cache/masks/biomedparse/a1b2c3d4e5f6.png\n"
            "  'right lung': 12.1% coverage, bbox=[0.08, 0.10, 0.45, 0.70]\n"
            "    Mask saved: cache/masks/biomedparse/b2c3d4e5f6a1.png\n"
            "  'lung opacity': 0.0% coverage, bbox=[0.00, 0.00, 0.00, 0.00]'"
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
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Text prompts describing what to segment (e.g., ['left lung', 'lung opacity']).",
                },
            },
            "required": ["image_path", "prompts"],
        }

    def run(self, image_path: str, prompts: list) -> str:
        resp = requests.post(
            f"{self.endpoint}/segment",
            json={"image_path": image_path, "prompts": prompts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        lines = ["BiomedParse Segmentation Results:"]
        for r in data["results"]:
            lines.append(f"  '{r['prompt']}': {r['coverage_pct']:.1f}% coverage, "
                        f"bbox=[{', '.join(f'{v:.2f}' for v in r['bbox'])}]")
            # Save mask to disk if available
            if r.get("mask_png_b64"):
                key = hashlib.md5(f"{image_path}_{r['prompt']}".encode()).hexdigest()[:12]
                mask_path = _MASK_DIR / f"{key}.png"
                mask_path.write_bytes(base64.b64decode(r["mask_png_b64"]))
                lines.append(f"    Mask saved: {mask_path}")
        return "\n".join(lines)
