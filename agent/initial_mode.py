"""
Initial-mode constants for A/B testing.

Contains the original system prompt, tool descriptions, user message, and
concept prior template from the initial commit (e498916). Used when
agent.prompt_mode == "initial" to faithfully reproduce the original agent
behavior while keeping infrastructure improvements (caching, 16-bit image
handling, trajectory tracking).
"""

# ─── System Prompt (original 8-line prompt from e498916) ─────────────────────

SYSTEM_PROMPT_INITIAL = """You are a radiology AI assistant. Generate a chest X-ray report using the available tools.

Output format — your final report MUST contain exactly these two sections:
  FINDINGS: Describe each observation with laterality, location, and severity.
  IMPRESSION: Summarize the key findings in 1-3 sentences.

Hard constraints:
- Only report findings supported by tool outputs or the concept prior. Do not invent findings.
- Never mention "compared to prior", "interval change", or prior studies — no prior study is provided in the current setup.
- Never fabricate patient history, clinical indication, or symptoms.
- If tools disagree on a finding, state the uncertainty rather than silently picking one.
- Replace specific measurements (e.g., "3.2 cm above the carina") with qualitative descriptions unless verified by FactCheXcker.
"""

# ─── Concept Prior Template (original wording) ──────────────────────────────

CONCEPT_PRIOR_TEMPLATE_INITIAL = """## CLEAR Concept Prior

Top matching observations from {num_concepts} MIMIC-CXR concepts (cosine similarity, not probability):

{concept_scores}
"""

# ─── User Message (original verbose instruction) ────────────────────────────

INITIAL_USER_MESSAGE = (
    "Please analyze this chest X-ray and generate a comprehensive radiology report. "
    "Use the available tools to gather information from specialized CXR models, "
    "then synthesize your findings into a final report with FINDINGS and IMPRESSION sections.\n\n"
    "When you have gathered sufficient information from the tools, stop calling tools "
    "and provide your final synthesized report directly."
)

# ─── Tool Descriptions (original 2-3 line descriptions from e498916) ────────

INITIAL_TOOL_DESCRIPTIONS = {
    "chexagent2_report": (
        "Generate a free-text radiology report for a chest X-ray using CheXagent-2 (3B VLM). "
        "Returns FINDINGS and IMPRESSION in natural radiologist-style prose."
    ),
    "chexagent2_srrg_report": (
        "Generate a structured radiology report organized by anatomical region "
        "(Lungs/Airways, Pleura, Cardiovascular, Other) using CheXagent-2-SRRG. "
        "Use this when findings span multiple anatomical categories."
    ),
    "chexone_report": (
        "Generate a radiology report using CheXOne (Qwen2.5-VL-3B), with optional "
        "step-by-step reasoning. Good as a second opinion when other models are "
        "ambiguous. Set reasoning=true for explicit clinical reasoning trace."
    ),
    "medversa_report": (
        "Generate a radiology report using MedVersa (7B generalist medical AI). "
        "Can incorporate patient context (age, gender, indication) if provided."
    ),
    "chexagent2_classify": (
        "Classify a chest X-ray using CheXagent-2. Supports three modes: "
        "'view' (PA/AP/Lateral view classification), "
        "'binary_disease' (yes/no for a specific disease like 'cardiomegaly'), "
        "'disease_id' (identify which diseases from a given list are present). "
        "Use this to cross-validate findings from report generation tools."
    ),
    "medversa_classify": (
        "Classify a chest X-ray for pathologies using MedVersa (7B). "
        "Returns diagnoses from 33 supported chest pathology categories. "
        "Use this for broad classification when you want to know ALL findings, "
        "not just check for a specific disease."
    ),
    "chexagent2_grounding": (
        "Visually ground a finding or phrase in a chest X-ray using CheXagent-2. "
        "Returns bounding box coordinates showing WHERE a finding is located. "
        "Supports: phrase grounding (any text), abnormality detection (specific disease), "
        "chest tube detection, rib fracture detection, and foreign objects detection. "
        "Use this to verify the spatial location of reported findings."
    ),
    "medversa_detect": (
        "Detect and localize abnormalities in a chest X-ray using MedVersa (7B). "
        "Returns bounding boxes around detected pathologies and structures."
    ),
    "chexagent2_vqa": (
        "Ask a specific question about a chest X-ray using CheXagent-2. "
        "Use this for targeted follow-up questions about findings, e.g., "
        "'Is there a pleural effusion on the left side?' or "
        "'What is the position of the endotracheal tube?'"
    ),
    "medversa_vqa": (
        "Ask a medical question about a chest X-ray using MedVersa (7B). "
        "Use for targeted follow-up questions, e.g., "
        "'Is the cardiac silhouette enlarged?' or 'Are there bilateral effusions?'"
    ),
    "biomedparse_segment": (
        "Segment anatomical structures or pathological findings in a chest X-ray "
        "using BiomedParse. Provide text prompts describing what to segment. "
        "Verified CXR prompts: 'left lung', 'right lung', 'lung', 'lung opacity', "
        "'viral pneumonia', 'COVID-19 infection'. "
        "Returns coverage percentage and bounding box for each segmented region. "
        "Use this to verify laterality, location, and extent of findings."
    ),
    "medsam3_segment": (
        "Segment findings in a chest X-ray using MedSAM3 (text-guided SAM). "
        "Supports broader vocabulary than BiomedParse — can segment pleural effusion, "
        "pneumothorax, consolidation, endotracheal tube, central venous catheter, etc. "
        "Returns coverage percentage and mask shape for the segmented region."
    ),
    "medversa_segment": (
        "Segment regions in a chest X-ray using MedVersa (7B) 2D segmentation. "
        "Returns pixel-level mask with coverage percentage. "
        "Use this to verify extent and location of findings."
    ),
    "factchexcker_verify": (
        "Verify and correct measurement hallucinations in a CXR report using FactCheXcker. "
        "Detects inaccurate quantifiable measurements (ETT position, tube placements) "
        "and corrects them based on the actual image. Only call this when the draft report "
        "contains specific numerical measurements about tubes, lines, or devices."
    ),
}
