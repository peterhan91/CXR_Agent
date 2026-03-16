"""FactCheXcker report verification tool."""

import requests
from tools.base import BaseCXRTool


class FactCheXckerVerifyTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8007"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "factchexcker_verify"

    @property
    def description(self) -> str:
        return (
            "[VERIFICATION] "
            "Verify and correct ETT-to-carina distance measurements in a CXR report using FactCheXcker. "
            "Uses CarinaNet+ (RetinaNet) to detect ETT tip and carina, then corrects distances. "
            "Scope: ETT measurements ONLY (78% of all report measurements). Cannot verify other tubes, "
            "lesion sizes, or cardiac-thoracic ratio. Reduces MAE from 1.93 to 0.82 cm. "
            "WHEN TO USE: Call ONLY when your draft report mentions ETT/endotracheal tube measurements. "
            "Do NOT call for reports without ETT — it will return 'no changes' instantly. "
            "EXAMPLE (changes needed): "
            "Input: {report: 'ETT tip is 2cm above the carina...'} → "
            "'FactCheXcker found measurement issues and corrected the report:\n"
            "ETT tip is 4.5cm above the carina...' "
            "EXAMPLE (no changes): "
            "'FactCheXcker: No measurement hallucinations detected. Report is consistent with the image.'"
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
                "report": {
                    "type": "string",
                    "description": "The draft radiology report to verify.",
                },
            },
            "required": ["image_path", "report"],
        }

    def run(self, image_path: str, report: str) -> str:
        resp = requests.post(
            f"{self.endpoint}/verify_report",
            json={"image_path": image_path, "report": report},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["changes_made"]:
            return (
                f"FactCheXcker found measurement issues and corrected the report:\n"
                f"{data['updated_report']}"
            )
        else:
            return "FactCheXcker: No measurement hallucinations detected. Report is consistent with the image."
