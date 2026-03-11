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
SYSTEM_PROMPT = """You are a radiology AI assistant. Generate a chest X-ray report using the available tools.

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


def build_skills_prompt(enabled_skills: list = None) -> str:
    """Load and assemble skill files into a single prompt block.

    Args:
        enabled_skills: List of skill filenames to load. If None, loads all
            skill files in default order.

    Returns:
        Combined skill text for injection into system prompt.
    """
    # No default skills — all clinical reasoning strategy is evolved via evotest.
    # Pass explicit filenames to load specific skills.
    skills_to_load = enabled_skills or []
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
