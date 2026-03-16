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
            "(Google 4B, Gemma 3 + MedSigLIP). "
            "Returns verbose paragraph-length answers with caveats and reasoning. "
            "WHEN TO USE: Use as a tiebreaker when CheXagent-2 VQA and classifiers disagree on a finding. "
            "Provides an independent third opinion from a different model family (Google vs Stanford). "
            "EXAMPLE: "
            "Input: {question: 'Is there a pleural effusion?'} → "
            "'MedGemma VQA:\nQ: Is there a pleural effusion?\n"
            "A: Based on the chest X-ray provided, there is no obvious pleural effusion. "
            "The costophrenic angles appear clear, and there is no blunting of the pleural surfaces.'"
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
            "Generate a radiology report using MedGemma (Google 4B, Gemma 3 + MedSigLIP). "
            "Output is markdown-formatted with **bold** headers and bullet points. "
            "RadGraph F1 27-30. 81% of reports judged sufficient by radiologist. "
            "WHEN TO USE: Call as a third-opinion report when chexagent2_report and chexone_report "
            "disagree on key findings. NOT needed for every study — only when the first two reports conflict. "
            "EXAMPLE OUTPUT: "
            "'MedGemma Report:\nThis is an AP upright chest X-ray.\n\n**Key Findings:**\n"
            "*   **Heart Size:** The heart appears to be mildly enlarged.\n"
            "*   **Lungs:** The lungs are clear, with no evidence of consolidation, pleural effusion, or pneumothorax.\n"
            "*   **Post-surgical changes:** There are surgical clips in the mediastinum.'"
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
