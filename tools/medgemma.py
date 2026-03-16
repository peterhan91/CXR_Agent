"""MedGemma VQA and report generation tools."""

import requests
from tools.base import BaseCXRTool


class MedGemmaVQATool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8010"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medgemma_vqa"

    @property
    def description(self) -> str:
        return (
            "[VQA] "
            "Ask a clinical question about a chest X-ray using MedGemma "
            "(Google's 4B medical vision-language model). "
            "WHEN TO USE: Use as a tiebreaker when CheXagent-2 VQA and classifiers disagree on a finding. "
            "Provides an independent third opinion from a different model family (Google vs Stanford). "
            "EXAMPLE: "
            "Input: {image_path: '...', question: 'Is there consolidation in the right lower lobe?'} → "
            "'MedGemma VQA:\nQ: Is there consolidation in the right lower lobe?\n"
            "A: There is no definite consolidation in the right lower lobe. There is minor atelectasis at the right base.'"
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
                "question": {
                    "type": "string",
                    "description": "Clinical question about the image.",
                },
            },
            "required": ["image_path", "question"],
        }

    def run(self, image_path: str, question: str) -> str:
        resp = requests.post(
            f"{self.endpoint}/vqa",
            json={"image_path": image_path, "question": question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"MedGemma VQA:\nQ: {question}\nA: {data['answer']}"


class MedGemmaReportTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8010"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medgemma_report"

    @property
    def description(self) -> str:
        return (
            "[REPORT GENERATOR] "
            "Generate a radiology report using MedGemma (Google's 4B medical "
            "vision-language model). "
            "WHEN TO USE: Call as a third-opinion report when chexagent2_report and chexone_report "
            "disagree on key findings. NOT needed for every study — only when the first two reports conflict. "
            "EXAMPLE OUTPUT: "
            "'MedGemma Report:\nThe cardiac silhouette is at the upper limits of normal. "
            "The lungs are clear bilaterally without consolidation, effusion, or pneumothorax. "
            "No acute osseous abnormalities.'"
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
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str) -> str:
        resp = requests.post(
            f"{self.endpoint}/generate_report",
            json={"image_path": image_path},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"MedGemma Report:\n{data['report']}"
