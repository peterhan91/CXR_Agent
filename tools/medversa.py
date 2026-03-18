"""MedVersa multi-task tool wrappers."""

import base64
import hashlib
from pathlib import Path

import requests
from tools.base import BaseCXRTool

_MASK_DIR = Path("cache/masks/medversa")
_MASK_DIR.mkdir(parents=True, exist_ok=True)


class MedVersaReportTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8004", legacy_mode: bool = False):
        self.endpoint = endpoint
        self.legacy_mode = legacy_mode

    @property
    def name(self) -> str:
        return "medversa_report"

    @property
    def description(self) -> str:
        return (
            "[REPORT GENERATOR] "
            "Generate a radiology report using MedVersa (LLaMA-2-7B, Harvard/Stanford). "
            "Outputs FINDINGS/IMPRESSION format. BLEU-4 17.8, RadCliQ 2.71. "
            "71% matched/exceeded human reports. Can incorporate patient context. "
            "WHEN TO USE: Call as an additional opinion when chexagent2 and chexone reports conflict. "
            "Particularly useful when patient context is available — pass age/gender/indication. "
            "EXAMPLE OUTPUT: "
            "'MedVersa Report:\nFindings:The patient is status post median sternotomy. "
            "The heart size is normal. The mediastinal and hilar contours are unremarkable. "
            "No focal consolidation, pleural effusion or pneumothorax is identified.\n"
            "Impression:No acute cardiopulmonary process.'"
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
        if self.legacy_mode:
            # Original payload: only image_path and context, no prompt/modality
            payload = {"image_path": image_path, "context": context}
        else:
            if not context:
                context = "Indication: Chest X-ray.\nComparison: None."
            payload = {
                "image_path": image_path,
                "context": context,
                "prompt": "Write a radiology report for <img0> with FINDINGS and IMPRESSION sections.",
                "modality": "cxr",
            }
        resp = requests.post(
            f"{self.endpoint}/generate_report",
            json=payload,
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
            "Covers 41 pathology categories (broader than CheXpert 14). F1 0.615 on 33 pathologies. "
            "WARNING: Endpoint is unreliable — may return segmentation tokens instead of labels. "
            "WHEN TO USE: Prefer chexzero_classify and cxr_foundation_classify instead. "
            "Only use this if you need pathologies beyond CheXpert-14 (e.g., hernia, scoliosis). "
            "EXAMPLE OUTPUT (when working): "
            "'MedVersa Classification (33 pathologies):\n  Cardiomegaly, Pleural Effusion, Atelectasis' "
            "EXAMPLE OUTPUT (broken): "
            "'MedVersa Classification (33 pathologies):\n  The segmentation mask of the cardiac silhouette is <2DSEG> .'"
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
            "Mean IoU 0.303 on CXR detection. "
            "WARNING: Endpoint often returns segmentation tokens instead of bounding boxes. "
            "WHEN TO USE: Prefer chexagent2_grounding for reliable bounding boxes. "
            "Only use this for open-ended abnormality detection when chexagent2 grounding is insufficient. "
            "EXAMPLE OUTPUT (broken): "
            "'MedVersa Detection:\n  The segmentation mask of bounding box is <2DSEG> .'"
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
            "Dice 0.955 on CheXmask organ segmentation. Returns coverage % and mask PNG. "
            "WHEN TO USE: Use to quantify the extent of a finding (e.g., 'how much lung is affected?'). "
            "Use biomedparse_segment for verified CXR prompts (lung, lung opacity). "
            "Use this for custom/unusual segmentation queries. "
            "EXAMPLE OUTPUT: "
            "'MedVersa 2D Segmentation:\n  Text: The segmentation mask of the abnormal region is <2DSEG> .\n"
            "  Coverage: 8.4% of image\n  Mask shape: [2544, 3056]\n"
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
            "WARNING: VQA endpoint is unreliable — often returns gibberish or truncated text. "
            "WHEN TO USE: Prefer chexagent2_vqa and medgemma_vqa for VQA. "
            "Only use this as a last resort when both other VQA tools are insufficient. "
            "EXAMPLE OUTPUT (broken): "
            "'MedVersa VQA:\n  Q: Is there a pleural effusion?\n  A: The PIC'"
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
