"""CheXzero zero-shot classification tool."""

import requests
from tools.base import BaseCXRTool


class CheXzeroClassifyTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8008"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexzero_classify"

    @property
    def description(self) -> str:
        return (
            "[CLASSIFIER] "
            "Zero-shot CXR classification using CheXzero (10-model ViT-B/32 CLIP ensemble). "
            "Classifies all 14 CheXpert pathologies in one call via majority vote. ~0.1s inference. "
            "Mean AUC 0.897 on CheXpert. Strongest: Edema (0.961), Pleural Effusion (0.958). "
            "Weakest: Atelectasis (0.798). May false-positive on Lung Lesion and Fracture. "
            "WHEN TO USE: Call this early (in parallel with chexagent2_report) as a systematic screen. "
            "The binary present/absent labels tell you what to look for in the reports and what to verify. "
            "EXAMPLE OUTPUT: "
            "'CheXzero 10-Model Ensemble (majority vote):\n"
            "  PRESENT: Atelectasis, Lung Lesion\n"
            "  ABSENT:  Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, "
            "Fracture, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, "
            "Pneumonia, Pneumothorax, Support Devices'"
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
                "pathologies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of pathologies to classify. Defaults to all 14 CheXpert labels: "
                        "Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, "
                        "Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, "
                        "Pleural Other, Pneumonia, Pneumothorax, Support Devices."
                    ),
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, pathologies: list = None) -> str:
        payload = {"image_path": image_path}
        if pathologies:
            payload["pathologies"] = pathologies
        resp = requests.post(
            f"{self.endpoint}/classify",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        preds = data["predictions"]

        present = [n for n, v in preds.items() if v == "present"]
        absent = [n for n, v in preds.items() if v == "absent"]
        lines = [f"CheXzero 10-Model Ensemble (majority vote):"]
        if present:
            lines.append(f"  PRESENT: {', '.join(present)}")
        if absent:
            lines.append(f"  ABSENT:  {', '.join(absent)}")
        return "\n".join(lines)
