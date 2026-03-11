"""MedSAM3 text-guided segmentation tool."""

import requests
from tools.base import BaseCXRTool


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
        return (
            f"MedSAM3 Segmentation for '{data['prompt']}':\n"
            f"  Coverage: {data['coverage_pct']:.1f}% of image\n"
            f"  Mask shape: {data['mask_shape']}"
        )
