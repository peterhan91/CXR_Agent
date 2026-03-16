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
            "Generate a radiology report using MedVersa (7B generalist medical AI). "
            "Can incorporate patient context (age, gender, indication) if provided."
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
            "Classify a chest X-ray for pathologies using MedVersa (7B). "
            "Returns diagnoses from 33 supported chest pathology categories. "
            "Use this for broad classification when you want to know ALL findings, "
            "not just check for a specific disease."
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
            "Detect and localize abnormalities in a chest X-ray using MedVersa (7B). "
            "Returns bounding boxes around detected pathologies and structures."
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
            "Segment regions in a chest X-ray using MedVersa (7B) 2D segmentation. "
            "Returns pixel-level mask with coverage percentage. "
            "Use this to verify extent and location of findings."
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
            "Ask a medical question about a chest X-ray using MedVersa (7B). "
            "Use for targeted follow-up questions, e.g., "
            "'Is the cardiac silhouette enlarged?' or 'Are there bilateral effusions?'"
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
