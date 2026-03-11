"""
System prompts for the CXR Report Generation ReAct Agent.

Adapted from mimic_skills CHAT_TEMPLATE pattern, redesigned for:
- CXR report generation (not clinical diagnosis)
- Anthropic native tool-use (not text-based ReAct parsing)
- CLEAR concept priors as structured input
- Skill-based clinical reasoning guidance (loaded from skills/ directory)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory containing skill markdown files
_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Default skill files loaded when no explicit list is provided
_DEFAULT_SKILLS = ["grounded_report.md"]


def _load_skill_file(filename: str) -> str:
    """Load a skill markdown file and strip YAML frontmatter."""
    path = _SKILLS_DIR / filename
    if not path.exists():
        logger.warning(f"Skill file not found: {path}")
        return ""
    text = path.read_text()
    # Strip YAML frontmatter (between --- markers)
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


# Fixed system prompt — contains ONLY hard constraints that evotest must never override.
# Clinical reasoning strategy (tool selection, interpretation, workflow) belongs in
# the evolved skill, not here.
SYSTEM_PROMPT = """You are a radiology AI assistant generating concise chest X-ray reports with grounded findings.

Output format — your final report MUST use this exact structure:

FINDINGS:
<one concise paragraph, 60-100 words, plain text, no markdown>

IMPRESSION:
<numbered list of 1-4 key findings>

GROUNDINGS:
<JSON array mapping key findings to bounding boxes from grounding tools>

Report style (CRITICAL for evaluation — match MIMIC-CXR format):
- Write plain text only. NO markdown formatting (no ##, no **, no bullets, no bold).
- Be concise: aim for 80-150 words total for FINDINGS + IMPRESSION combined.
- Use standard radiology phrasing (e.g., "There is...", "No evidence of...", "The cardiac silhouette is...").
- Do NOT mention tool names, model names, or "concept prior" in report text.
- Do NOT discuss model agreement or disagreement in report text.
- Only report findings supported by tool outputs or the concept prior. Do not invent findings.
- Never mention "compared to prior", "interval change", or prior studies.
- Never fabricate patient history, clinical indication, or symptoms.
- Replace specific measurements with qualitative descriptions unless verified.

Grounding:
- For each key abnormal finding, use a grounding tool to get its bounding box.
- Include grounding results in the GROUNDINGS section as JSON array.
- Format: [{"finding": "<phrase>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>"}]
"""


def build_skills_prompt(enabled_skills: list = None) -> str:
    """Load and assemble skill files into a single prompt block.

    Args:
        enabled_skills: List of skill filenames to load. If None, loads
            default skill files.

    Returns:
        Combined skill text for injection into system prompt.
    """
    skills_to_load = enabled_skills if enabled_skills is not None else _DEFAULT_SKILLS
    skill_texts = []

    for filename in skills_to_load:
        text = _load_skill_file(filename)
        if text:
            skill_texts.append(text)

    if not skill_texts:
        return ""

    return "\n\n---\n\n".join(skill_texts)


CONCEPT_PRIOR_TEMPLATE = """## CLEAR Concept Prior

Top matching observations from {num_concepts} MIMIC-CXR concepts (cosine similarity, not probability):

{concept_scores}
"""

SKILL_INJECTION_TEMPLATE = """## Clinical Reasoning Skill (Evolved)

{skill_text}
"""
