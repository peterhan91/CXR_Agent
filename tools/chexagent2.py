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
            "Generate a free-text radiology report using CheXagent-2 (3B, Stanford). "
            "Output uses category tags like [Breathing: Lungs] and **bold** for abnormals. "
            "Trained on CheXinstruct (6.1M triplets, 28 CXR datasets). F1CheXbert avg 55.2. "
            "Known to fabricate ETT distance measurements. "
            "WHEN TO USE: Call this first for every study to get a baseline narrative report. "
            "EXAMPLE OUTPUT: "
            "'CheXagent-2 Report:\n"
            "[Breathing: Lungs] The lungs are clear without focal consolidation. "
            "[Breathing: Pleura] No pleural effusion or pneumothorax is seen. "
            "[Cardiac: Heart Size and Borders] The cardiac and mediastinal silhouettes are stable. "
            "[Everything else: Bones] No displaced fracture is seen.'"
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
            "Generate a structured report organized by anatomical region using CheXagent-2-SRRG. "
            "Regions: Lungs and Airways, Pleura, Cardiovascular, Hila and Mediastinum, Musculoskeletal. "
            "WHEN TO USE: Call alongside chexagent2_report to get region-by-region findings — "
            "this helps you avoid missing findings in specific anatomical areas. "
            "EXAMPLE OUTPUT: "
            "'CheXagent-2-SRRG Structured Findings:\n"
            "Lungs and Airways:\n- Low lung volumes\n- No focal consolidation\n- No pneumothorax\n\n"
            "Pleura:\n- No pleural effusion\n\n"
            "Cardiovascular:\n- Normal heart size\n\n"
            "Hila and Mediastinum:\n- Normal mediastinal contours\n- Presence of midline sternotomy wires and mediastinal clips\n\n"
            "Musculoskeletal and Chest Wall:\n- No bony abnormalities detected'"
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
            "Returns bounding box coordinates [0-1] showing WHERE a finding is located. "
            "Supports: phrase_grounding (any text), abnormality (specific disease), "
            "chest_tube, rib_fracture, foreign_objects. "
            "Returns 'No X detected.' with empty boxes when finding is absent. "
            "WHEN TO USE: After you have confirmed a finding exists, call this to get its spatial location "
            "for the GROUNDINGS section. Use task='phrase_grounding' + phrase for specific findings. "
            "EXAMPLE (phrase grounding, finding present): "
            "Input: {image_path: '...', task: 'phrase_grounding', phrase: 'pleural effusion'} → "
            "'CheXagent-2 Grounding (phrase_grounding):\n  Response: pleural effusion\n"
            "  Box 1: x=[0.050, 0.100] y=[0.680, 0.764]\n"
            "  Box 2: x=[0.870, 0.940] y=[0.680, 0.764]' "
            "EXAMPLE (abnormality, finding absent): "
            "Input: {image_path: '...', task: 'abnormality', disease_name: 'cardiomegaly'} → "
            "'CheXagent-2 Grounding (abnormality):\n  Response: No cardiomegaly detected.\n"
            "  No bounding boxes detected.'"
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
            "Classify a chest X-ray using CheXagent-2. Three modes: "
            "'view' (PA/AP/Lateral), "
            "'binary_disease' (yes/no for one disease), "
            "'disease_id' (which diseases from a list are present — returns only the present ones). "
            "WHEN TO USE: Use binary_disease to confirm/deny a specific finding when classifiers disagree. "
            "Use disease_id to check multiple diseases at once. "
            "EXAMPLE (disease_id): "
            "Input: {image_path: '...', task: 'disease_id', disease_names: ['pneumonia', 'atelectasis', 'cardiomegaly', 'pleural effusion']} → "
            "'CheXagent-2 Classification (disease_id):\n  Atelectasis' "
            "EXAMPLE (view): "
            "Input: {image_path: '...', task: 'view'} → "
            "'CheXagent-2 Classification (view):\n  (b) AP'"
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
            "Returns very short answers (often 1-3 words for yes/no questions). "
            "For open-ended questions, may return tagged prose like reports. "
            "WHEN TO USE: Use for targeted follow-up when you need to resolve ambiguity — "
            "e.g., laterality ('left or right?'), severity ('small or large?'), or device identification. "
            "Do NOT use for broad screening — use classifiers instead. "
            "EXAMPLE: "
            "Input: {question: 'Is there a pleural effusion?'} → "
            "'CheXagent-2 VQA:\n  Q: Is there a pleural effusion?\n  A: No' "
            "EXAMPLE: "
            "Input: {question: 'What devices or lines are present?'} → "
            "'CheXagent-2 VQA:\n  Q: What devices or lines are present?\n  A: Cabg grafts'"
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
