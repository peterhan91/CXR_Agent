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
            "Ask a clinical question about a chest X-ray using MedGemma "
            "(Google's 4B medical vision-language model). Use for follow-up "
            "questions, verification of specific findings, or when other "
            "tools disagree."
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
            "Generate a radiology report using MedGemma (Google's 4B medical "
            "vision-language model). Provides a third-opinion report alongside "
            "CheXagent-2 and CheXOne."
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
