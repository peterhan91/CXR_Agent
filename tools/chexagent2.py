"""CheXagent-2 multi-task tool wrappers."""

import requests
from tools.base import BaseCXRTool


class CheXagent2ReportTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexagent2_report"

    @property
    def description(self) -> str:
        return (
            "[REPORT GENERATOR] "
            "Generate a free-text radiology report for a chest X-ray using CheXagent-2 (3B VLM). "
            "Returns FINDINGS and IMPRESSION in natural radiologist-style prose. "
            "WHEN TO USE: Call this first for every study to get a baseline narrative report. "
            "EXAMPLE OUTPUT: "
            "'CheXagent-2 Report:\nFINDINGS: The heart is mildly enlarged. There is a small left pleural effusion. "
            "Mild bibasilar atelectasis is noted. No focal consolidation or pneumothorax. "
            "A right-sided PICC line terminates in the lower SVC.\n"
            "IMPRESSION: Mild cardiomegaly with small left pleural effusion and bibasilar atelectasis.'"
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
        return f"CheXagent-2 Report:\n{resp.json()['report']}"


class CheXagent2SRRGTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexagent2_srrg_report"

    @property
    def description(self) -> str:
        return (
            "[REPORT GENERATOR] "
            "Generate a structured radiology report organized by anatomical region "
            "(Lungs/Airways, Pleura, Cardiovascular, Other) using CheXagent-2-SRRG. "
            "WHEN TO USE: Call alongside chexagent2_report to get region-by-region findings — "
            "this helps you avoid missing findings in specific anatomical areas. "
            "EXAMPLE OUTPUT: "
            "'CheXagent-2-SRRG Structured Findings:\n"
            "Lungs and Airways: Bibasilar atelectasis. No focal consolidation.\n"
            "Pleura: Small left-sided pleural effusion. No pneumothorax.\n"
            "Cardiovascular: The heart is mildly enlarged. Mediastinal contours are normal.\n"
            "Other: Right PICC line with tip in the lower SVC. No acute osseous abnormality.'"
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
            f"{self.endpoint}/generate_srrg",
            json={"image_path": image_path},
            timeout=120,
        )
        resp.raise_for_status()
        return f"CheXagent-2-SRRG Structured Findings:\n{resp.json()['report']}"


class CheXagent2GroundingTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexagent2_grounding"

    @property
    def description(self) -> str:
        return (
            "[GROUNDING] "
            "Visually ground a finding or phrase in a chest X-ray using CheXagent-2. "
            "Returns bounding box coordinates showing WHERE a finding is located. "
            "Supports: phrase grounding (any text), abnormality detection (specific disease), "
            "chest tube detection, rib fracture detection, and foreign objects detection. "
            "WHEN TO USE: After you have confirmed a finding exists, call this to get its spatial location "
            "for the GROUNDINGS section. Use task='phrase_grounding' + phrase for specific findings, "
            "or task='abnormality' + disease_name for disease localization. "
            "EXAMPLE (phrase grounding): "
            "Input: {image_path: '...', task: 'phrase_grounding', phrase: 'pleural effusion'} → "
            "'CheXagent-2 Grounding (phrase_grounding):\n  Response: pleural effusion\n"
            "  Box 1: x=[0.521, 0.943] y=[0.612, 0.891]' "
            "EXAMPLE (abnormality): "
            "Input: {image_path: '...', task: 'abnormality', disease_name: 'cardiomegaly'} → "
            "'CheXagent-2 Grounding (abnormality):\n  Response: cardiomegaly\n"
            "  Box 1: x=[0.250, 0.750] y=[0.300, 0.800]'"
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
                "task": {
                    "type": "string",
                    "enum": ["phrase_grounding", "abnormality", "chest_tube", "rib_fracture", "foreign_objects"],
                    "description": "Type of grounding task.",
                    "default": "phrase_grounding",
                },
                "phrase": {
                    "type": "string",
                    "description": "Text phrase to locate (for phrase_grounding task).",
                },
                "disease_name": {
                    "type": "string",
                    "description": "Disease to detect (for abnormality task).",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, task: str = "phrase_grounding",
            phrase: str = None, disease_name: str = None) -> str:
        resp = requests.post(
            f"{self.endpoint}/ground",
            json={
                "image_path": image_path,
                "task": task,
                "phrase": phrase,
                "disease_name": disease_name,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        lines = [f"CheXagent-2 Grounding ({task}):"]
        lines.append(f"  Response: {data['result']}")
        if data["boxes"]:
            for i, box in enumerate(data["boxes"]):
                lines.append(
                    f"  Box {i+1}: x=[{box['x_min']:.3f}, {box['x_max']:.3f}] "
                    f"y=[{box['y_min']:.3f}, {box['y_max']:.3f}]"
                )
        else:
            lines.append("  No bounding boxes detected.")
        return "\n".join(lines)


class CheXagent2ClassifyTool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexagent2_classify"

    @property
    def description(self) -> str:
        return (
            "[CLASSIFIER] "
            "Classify a chest X-ray using CheXagent-2. Supports three modes: "
            "'view' (PA/AP/Lateral view classification), "
            "'binary_disease' (yes/no for a specific disease like 'cardiomegaly'), "
            "'disease_id' (identify which diseases from a given list are present). "
            "WHEN TO USE: Use binary_disease to confirm/deny a specific finding when classifiers disagree. "
            "Use disease_id to check multiple specific diseases at once. "
            "EXAMPLE (binary_disease): "
            "Input: {image_path: '...', task: 'binary_disease', disease_name: 'cardiomegaly'} → "
            "'CheXagent-2 Classification (binary_disease):\n  Yes, cardiomegaly is present.' "
            "EXAMPLE (disease_id): "
            "Input: {image_path: '...', task: 'disease_id', disease_names: ['pneumonia', 'atelectasis', 'effusion']} → "
            "'CheXagent-2 Classification (disease_id):\n  atelectasis, effusion'"
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
                "task": {
                    "type": "string",
                    "enum": ["view", "binary_disease", "disease_id"],
                    "description": "Classification task type.",
                    "default": "binary_disease",
                },
                "disease_name": {
                    "type": "string",
                    "description": "Disease to check (for binary_disease task).",
                },
                "disease_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of diseases to identify from (for disease_id task).",
                },
            },
            "required": ["image_path"],
        }

    def run(self, image_path: str, task: str = "binary_disease",
            disease_name: str = None, disease_names: list = None) -> str:
        resp = requests.post(
            f"{self.endpoint}/classify",
            json={
                "image_path": image_path,
                "task": task,
                "disease_name": disease_name,
                "disease_names": disease_names,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"CheXagent-2 Classification ({task}):\n  {data['result']}"


class CheXagent2VQATool(BaseCXRTool):

    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return "chexagent2_vqa"

    @property
    def description(self) -> str:
        return (
            "[VQA] "
            "Ask a specific question about a chest X-ray using CheXagent-2. "
            "WHEN TO USE: Use for targeted follow-up when you need to resolve ambiguity — "
            "e.g., laterality ('left or right?'), severity ('small or large?'), or device position. "
            "Do NOT use for broad screening — use classifiers instead. "
            "EXAMPLE: "
            "Input: {image_path: '...', question: 'Is the pleural effusion on the left or right side?'} → "
            "'CheXagent-2 VQA:\n  Q: Is the pleural effusion on the left or right side?\n  A: Left side.' "
            "EXAMPLE: "
            "Input: {image_path: '...', question: 'What is the position of the endotracheal tube?'} → "
            "'CheXagent-2 VQA:\n  Q: What is the position of the endotracheal tube?\n  A: The endotracheal tube tip is approximately 4 cm above the carina.'"
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
                    "description": "Question to ask about the image.",
                },
            },
            "required": ["image_path", "question"],
        }

    def run(self, image_path: str, question: str) -> str:
        resp = requests.post(
            f"{self.endpoint}/vqa",
            json={"image_path": image_path, "question": question},
            timeout=60,
        )
        resp.raise_for_status()
        return f"CheXagent-2 VQA:\n  Q: {question}\n  A: {resp.json()['result']}"
