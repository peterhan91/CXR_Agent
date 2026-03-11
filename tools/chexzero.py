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
            "Zero-shot chest X-ray classification using CheXzero (CLIP fine-tuned on MIMIC-CXR). "
            "Classifies an image for multiple pathologies simultaneously using positive/negative "
            "prompt pairs (e.g., 'Atelectasis' vs 'no Atelectasis') with calibrated probabilities. "
            "Default: classifies all 14 CheXpert pathologies in one call. "
            "Returns probability [0-1] for each pathology. Higher = more likely present. "
            "Use this for comprehensive multi-label screening — much faster than testing one disease at a time."
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

        lines = ["CheXzero Zero-Shot Classification (probability of each finding):"]
        # Sort by probability descending
        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_preds:
            marker = "+" if prob > 0.5 else "-"
            lines.append(f"  [{marker}] {name}: {prob:.3f}")
        return "\n".join(lines)
