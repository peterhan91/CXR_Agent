"""MedVersa multi-task tool wrappers."""

import base64
import hashlib
from pathlib import Path

import requests
from tools.base import BaseCXRTool

_MASK_DIR = Path("cache/masks/medversa")
_MASK_DIR.mkdir(parents=True, exist_ok=True)


class MedVersaReportTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medversa_report"

    @property
    def description(self) -> str:
        return (
            "[REPORT GENERATOR] "
            "Generate a radiology report using MedVersa (7B generalist medical AI). "
            "Can incorporate patient context (age, gender, indication) if provided. "
            "WHEN TO USE: Call as an additional opinion when chexagent2 and chexone reports conflict. "
            "Particularly useful when patient context is available — pass age/gender/indication for more targeted report. "
            "EXAMPLE OUTPUT: "
            "'MedVersa Report:\nFINDINGS: The heart is normal in size. The lungs are clear. "
            "There is no pleural effusion or pneumothorax. The mediastinal contours are normal.\n"
            "IMPRESSION: No acute cardiopulmonary abnormality.'"
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
                "context": {
                    "type": "string",
                    "description": "Optional patient context (e.g., 'Age:65. Gender:M. Indication: shortness of breath').",
                    "default": "",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, context: str = "") -> str:
        if not context:
            context = "Indication: Chest X-ray.\nComparison: None."
        resp = requests.post(
            f"{self.endpoint}/generate_report",
            json={
                "image_path": image_path,
                "context": context,
                "prompt": "Write a radiology report for <img0> with FINDINGS and IMPRESSION sections.",
                "modality": "cxr",
            },
            timeout=180,
        )
        resp.raise_for_status()
        return f"MedVersa Report:\n{resp.json()['report']}"


class MedVersaClassifyTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medversa_classify"

    @property
    def description(self) -> str:
        return (
            "[CLASSIFIER] "
            "Classify a chest X-ray for pathologies using MedVersa (7B). "
            "Returns diagnoses from 33 supported chest pathology categories (broader than CheXpert 14). "
            "WHEN TO USE: Call when you want to check for pathologies beyond the 14 CheXpert labels "
            "(e.g., hernia, subcutaneous emphysema, mediastinal widening). "
            "EXAMPLE OUTPUT: "
            "'MedVersa Classification (33 pathologies):\n  Cardiomegaly, Pleural Effusion, Atelectasis'"
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
                "context": {
                    "type": "string",
                    "description": "Optional patient context.",
                    "default": "",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, context: str = "") -> str:
        resp = requests.post(
            f"{self.endpoint}/classify",
            json={"image_path": image_path, "context": context},
            timeout=120,
        )
        resp.raise_for_status()
        return f"MedVersa Classification (33 pathologies):\n  {resp.json()['result']}"


class MedVersaDetectTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medversa_detect"

    @property
    def description(self) -> str:
        return (
            "[GROUNDING] "
            "Detect and localize abnormalities in a chest X-ray using MedVersa (7B). "
            "Returns bounding boxes around detected pathologies and structures. "
            "WHEN TO USE: Use for open-ended abnormality detection — finds things you might not have asked about. "
            "Different from chexagent2_grounding which requires you to specify what to look for. "
            "EXAMPLE OUTPUT: "
            "'MedVersa Detection:\n  Detected: cardiomegaly [0.22, 0.28, 0.78, 0.85], "
            "pleural effusion [0.55, 0.60, 0.95, 0.95]'"
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
                    "description": "Detection prompt (e.g., 'Can you detect any abnormality on <img0>?').",
                    "default": "Can you detect any abnormality on <img0>? Please output the bounding box.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional patient context.",
                    "default": "",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, prompt: str = None, context: str = "") -> str:
        payload = {"image_path": image_path, "context": context}
        if prompt:
            payload["prompt"] = prompt
        resp = requests.post(
            f"{self.endpoint}/detect",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return f"MedVersa Detection:\n  {resp.json()['result']}"


class MedVersaSegmentTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medversa_segment"

    @property
    def description(self) -> str:
        return (
            "[GROUNDING] "
            "Segment regions in a chest X-ray using MedVersa (7B) 2D segmentation. "
            "Returns pixel-level mask with coverage percentage. "
            "WHEN TO USE: Use to quantify the extent of a finding (e.g., 'how much lung is affected?'). "
            "Similar to biomedparse_segment but uses a different model — use biomedparse for verified CXR prompts, "
            "use this for custom/unusual segmentation queries. "
            "EXAMPLE OUTPUT: "
            "'MedVersa 2D Segmentation:\n  Text: lung opacity segmented\n"
            "  Coverage: 12.3% of image\n  Mask shape: [512, 512]\n"
            "  Mask saved: cache/masks/medversa/a1b2c3d4e5f6.png'"
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
                    "description": "What to segment (e.g., 'Can you segment the lung region in <img0>?').",
                    "default": "Can you segment the abnormal region in <img0>?",
                },
                "context": {
                    "type": "string",
                    "description": "Optional patient context.",
                    "default": "",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, prompt: str = None, context: str = "") -> str:
        payload = {"image_path": image_path, "context": context}
        if prompt:
            payload["prompt"] = prompt
        resp = requests.post(
            f"{self.endpoint}/segment_2d",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        lines = ["MedVersa 2D Segmentation:"]
        lines.append(f"  Text: {data['result']}")
        if data["has_mask"]:
            lines.append(f"  Coverage: {data['coverage_pct']:.1f}% of image")
            lines.append(f"  Mask shape: {data['mask_shape']}")
            # Save mask to disk if available
            if data.get("mask_png_b64"):
                key = hashlib.md5(f"{image_path}_{prompt or 'default'}".encode()).hexdigest()[:12]
                mask_path = _MASK_DIR / f"{key}.png"
                mask_path.write_bytes(base64.b64decode(data["mask_png_b64"]))
                lines.append(f"  Mask saved: {mask_path}")
        else:
            lines.append("  No segmentation mask produced.")
        return "\n".join(lines)


class MedVersaVQATool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "medversa_vqa"

    @property
    def description(self) -> str:
        return (
            "[VQA] "
            "Ask a medical question about a chest X-ray using MedVersa (7B). "
            "WHEN TO USE: Use as an additional VQA opinion when chexagent2_vqa and medgemma_vqa disagree, "
            "or when you need a third verification source. "
            "EXAMPLE: "
            "Input: {image_path: '...', question: 'Is the cardiac silhouette enlarged?'} → "
            "'MedVersa VQA:\n  Q: Is the cardiac silhouette enlarged?\n  A: Yes, the cardiac silhouette appears mildly enlarged.'"
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
                    "description": "Medical question about the image.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional patient context.",
                    "default": "",
                },
            },
            "required": ["image_path", "question"],
        }

    def run(self, image_path: str, question: str, context: str = "") -> str:
        resp = requests.post(
            f"{self.endpoint}/vqa",
            json={"image_path": image_path, "question": question, "context": context},
            timeout=120,
        )
        resp.raise_for_status()
        return f"MedVersa VQA:\n  Q: {question}\n  A: {resp.json()['result']}"
