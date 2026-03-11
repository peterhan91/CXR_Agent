"""CheXOne report generation tool."""

import requests
from tools.base import BaseCXRTool


class CheXOneReportTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8002"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexone_report"

    @property
    def description(self) -> str:
        return (
            "Generate a radiology report using CheXOne (Qwen2.5-VL-3B), with optional "
            "step-by-step reasoning. Good as a second opinion when other models are "
            "ambiguous. Set reasoning=true for explicit clinical reasoning trace."
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
                "reasoning": {
                    "type": "boolean",
                    "description": "If true, model shows step-by-step reasoning before the final answer.",
                    "default": False,
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, reasoning: bool = False) -> str:
        resp = requests.post(
            f"{self.endpoint}/generate_report",
            json={"image_path": image_path, "reasoning": reasoning},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        mode = "Reasoning" if reasoning else "Instruct"
        return f"CheXOne Report ({mode} mode):\n{data['report']}"
