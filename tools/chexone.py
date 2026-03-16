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
            "[REPORT GENERATOR] "
            "Generate a radiology report using CheXOne (Qwen2.5-VL-3B, Stanford). "
            "Trained with GRPO RL on CheXInstruct-v2. ReXrank top-3 on ReXGradient, top-2 on VQA. "
            "WHEN TO USE: Call this as a second-opinion report alongside chexagent2_report. "
            "Compare both to identify agreements (high confidence) and disagreements (need verification). "
            "Set reasoning=true only when you need to understand WHY a finding was reported (~2x slower). "
            "EXAMPLE OUTPUT (reasoning=false): "
            "'CheXOne Report (Instruct mode):\n"
            "The sternotomy wires are stable. The lung volumes remain low. "
            "There is no evidence of acute disease. There is no pneumonia. "
            "There is no pleural effusion or pneumothorax identified. The cardiac silhouette is normal.' "
            "EXAMPLE OUTPUT (reasoning=true): "
            "'CheXOne Report (Reasoning mode):\n"
            "**Thinking Process:**\n1. **Assess Sternotomy Wires**: The sternotomy wires appear unchanged...\n"
            "2. **Evaluate Lung Volumes**: Both lung fields show reduced air trapping...\n"
            "Final report: ...'"
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
