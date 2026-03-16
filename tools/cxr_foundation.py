"""CXR Foundation (Google ELIXR v2) zero-shot classification tool."""

import requests
from tools.base import BaseCXRTool


class CXRFoundationClassifyTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8008"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "cxr_foundation_classify"

    @property
    def description(self) -> str:
        return (
            "[CLASSIFIER] "
            "Zero-shot CXR classification using Google CXR Foundation (ELIXR v2, EfficientNet-L2 + BERT). "
            "Independent from CheXzero — different architecture, different training data (821K CXRs). "
            "Mean AUC 0.850 (13 findings). More conservative than CheXzero (fewer false positives). "
            "WHEN TO USE: Call in parallel with chexzero_classify. Compare both outputs: "
            "if both agree a finding is PRESENT → high confidence. "
            "If they disagree → use chexagent2_classify or VQA to break the tie. "
            "EXAMPLE OUTPUT: "
            "'CXR Foundation Zero-Shot Classification:\n"
            "  ABSENT:  Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, "
            "Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, "
            "Pleural Other, Pneumonia, Pneumothorax, Support Devices'"
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
            timeout=120,  # CXR Foundation is slower (CPU mode)
        )
        resp.raise_for_status()
        data = resp.json()
        preds = data["predictions"]

        present = [n for n, v in preds.items() if v == "present"]
        absent = [n for n, v in preds.items() if v == "absent"]
        lines = ["CXR Foundation Zero-Shot Classification:"]
        if present:
            lines.append(f"  PRESENT: {', '.join(present)}")
        if absent:
            lines.append(f"  ABSENT:  {', '.join(absent)}")
        return "\n".join(lines)
