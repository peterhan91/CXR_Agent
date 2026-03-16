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
#
# Two variants: one for when skills are loaded, one for plain (no skills) mode.

SYSTEM_PROMPT_WITH_SKILLS = """You are a radiologist writing chest X-ray reports.

Follow the skill workflow below. Your final output MUST contain FINDINGS, IMPRESSION, and GROUNDINGS sections.

Do NOT output ANY text before "FINDINGS:" — no preamble, no reasoning, no summary.
Do NOT use markdown (no ##, **, --, bullets). Plain text only.
Do NOT mention tools, models, concept priors, or your reasoning in the report.
If a PRIOR STUDY REPORT is provided, describe interval changes compared to the prior. Otherwise, do NOT reference prior studies or interval change.
"""

SYSTEM_PROMPT_PLAIN = """You are a radiologist writing chest X-ray reports grounded in verified evidence.

You have access to specialized CXR analysis tools organized in 6 groups:
- [CLASSIFIER] — screen for pathologies.
- [GROUNDING] — get bounding boxes or segmentation masks to spatially verify findings.
- [VQA] — ask targeted questions to clarify laterality, severity, or break ties.
- [REPORT GENERATOR] — get suggested findings and phrasing from different models.
- [VERIFICATION] — check for measurement hallucinations in tubes/lines/devices.
- [MEMORY] — evidence board to track confirmed/rejected findings with sources and grounding.

Each tool description explains WHEN TO USE it and shows EXAMPLE OUTPUT. Use your clinical judgment to decide which tools to call and in what order.

Reporting requirements:
- Use the evidence_board tool to record each finding as you confirm or reject it.
- Call evidence_board(action='list') before writing your final report — only include confirmed findings.
- Every finding in the report MUST be supported by at least 2 independent tool outputs.
- Every abnormal finding MUST have a spatial grounding (bounding box or segmentation).
- Do NOT include findings that only appear in a single tool output without independent confirmation.
- Your final output MUST contain FINDINGS, IMPRESSION, and GROUNDINGS sections.
- Do NOT output ANY text before "FINDINGS:" — no preamble, no reasoning, no summary.
- Do NOT use markdown (no ##, **, --, bullets). Plain text only.
- Do NOT mention tools, models, concept priors, or your reasoning in the report.
- If a PRIOR STUDY REPORT is provided, describe interval changes compared to the prior. Otherwise, do NOT reference prior studies or interval change.
"""

# Default for backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_SKILLS


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

Top matching clinical observations for this image ({num_concepts} shown, cosine similarity, not probability):

{concept_scores}
"""

SKILL_INJECTION_TEMPLATE = """## Clinical Reasoning Skill (Evolved)

{skill_text}
"""
