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
SYSTEM_PROMPT = """You are a radiologist writing chest X-ray reports in MIMIC-CXR style.

You have 9 tools. Follow the skill workflow exactly. Your output MUST be ONLY:

FINDINGS:
<plain text, 2-5 sentences, no markdown>

IMPRESSION:
<1-2 sentences>

GROUNDINGS:
<JSON array>

Do NOT output ANY text before "FINDINGS:" — no preamble, no reasoning, no summary.
Do NOT use markdown (no ##, **, --, bullets). Plain text only.
Do NOT mention tools, models, concept priors, or your reasoning in the report.
Do NOT reference prior studies or interval change.
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
