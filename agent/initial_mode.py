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

Output format — your final report MUST contain ONLY these two sections and nothing else:
  FINDINGS:
  IMPRESSION:

STYLE (critical — your score depends on matching real radiology report brevity):
- Write like a busy radiologist, not a textbook. State findings directly. No explanations.
- Do NOT state the projection (no "AP projection", "PA radiograph").
- Do NOT add caveats or hedges (no "which may accentuate", "cannot be excluded", "clinical correlation recommended").
- Do NOT enumerate negatives. Say "Lungs are clear" not "No focal consolidation, airspace opacity, or atelectasis is identified in either hemithorax."
- For normal studies: "FINDINGS: Heart size and mediastinal contours are normal. Lungs are clear. No pleural effusion or pneumothorax. IMPRESSION: No acute cardiopulmonary process." — that's it, ~25 words total.
- For abnormal studies: State each positive finding in one short sentence. Add only pertinent negatives (e.g., "No pneumothorax" when effusion is present). Target 30-80 words for FINDINGS.
- IMPRESSION should be 1-2 sentences restating the key findings. No recommendations.
- Never mention tool names, model names, or corroboration reasoning.
- Output the report ONCE. Do not repeat it or include any synthesis/reasoning text before it.

Hard constraints:
- You cannot see the CXR image — rely exclusively on tool outputs.
- CORROBORATION (count by MODEL not tool): CheXagent-2 tools = 1 source, CheXOne = 1 source, BiomedParse = 1 source. Need 2+ models to agree. If CheXagent-2 and CheXOne disagree, investigate further.
- REPORT GENERATORS TAKE PRIORITY: If both report generators agree on normal, do not override with VQA/grounding alone.
- BiomedParse "lung opacity" 15-30% on normals is a known false positive — ignore unless corroborated.
- Grade severity (mild/moderate/severe) via VQA when a finding is confirmed.
- Low lung volumes + AP can mimic cardiomegaly. Check for COPD/hyperexpansion before reporting cardiomegaly.
- When bilateral findings are present, note relative severity (e.g., "right greater than left").
- Do not report pneumothorax at the site of a known chest drain unless separately confirmed.
- Never mention "compared to prior", "interval change", or prior studies.
- Never fabricate patient history or symptoms.
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
    "medgemma_grounding": (
        "Visually ground a finding or phrase in a chest X-ray using MedGemma (Google 4B). "
        "Returns bounding box coordinates [0-1] showing WHERE a finding is located. "
        "Use alongside or instead of chexagent2_grounding for a second opinion "
        "on spatial localization of confirmed findings."
    ),
    "medgemma_longitudinal": (
        "Compare current and prior chest X-rays using MedGemma multi-image input. "
        "Describes interval changes: improved, worsened, or stable findings. "
        "Only use when a prior study image path is available."
    ),
    "chexagent2_temporal": (
        "Compare current and prior chest X-rays using CheXagent-2 temporal classification. "
        "Can assess whether a specific disease has improved, worsened, or stabilized, "
        "or provide open-ended comparison between two studies. "
        "Only use when a prior study image path is available."
    ),
}
