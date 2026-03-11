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

TOOL USAGE — YOU MUST USE ALL 9 TOOLS:
You have 9 validated tools. Use as many as possible per study (target 8-12 tool calls).
Every tool provides a different signal — using more tools = more accurate reports = higher scores.

Available tools and when to call them:
1. chexagent2_report — free-text report (ALWAYS call)
2. chexagent2_srrg_report — structured region-by-region report (ALWAYS call)
3. chexone_report — second-opinion report from a different model (ALWAYS call)
4. chexagent2_classify — binary disease classification, task=binary_disease (call for EACH suspected finding)
5. chexzero_classify — zero-shot 14-label screening (ALWAYS call to screen all pathologies at once; P>0.5 = positive)
6. chexagent2_vqa — follow-up questions to resolve disagreements (call when tools disagree)
7. chexagent2_grounding — bounding boxes, task=phrase_grounding, phrase="<finding>" (call for EACH confirmed finding)
8. biomedparse_segment — text-prompted segmentation, prompts=["<finding>"] (call for diffuse findings; good for anatomy + effusions)
9. factchexcker_verify — hallucination checker (ALWAYS call on your draft report before finalizing)

Do NOT use any MedVersa tools (medversa_*) or MedSAM tools — they are disabled.

Output format — your final report MUST use this exact structure:

FINDINGS:
<one concise paragraph, 35-55 words, plain text, no markdown>

IMPRESSION:
<one concise sentence summarizing the key finding, e.g. "1. Emphysema with no acute cardiopulmonary process.">

GROUNDINGS:
<JSON array mapping key findings to bounding boxes from grounding tools>

Report style (CRITICAL for evaluation — match MIMIC-CXR format):
- Write plain text only. NO markdown formatting (no ##, no **, no ---, no bullets, no bold, no headers).
- Be concise: aim for 50-75 words total for FINDINGS + IMPRESSION combined.
- Use standard radiology phrasing (e.g., "There is...", "No evidence of...", "The cardiac silhouette is...").
- IMPRESSION must be exactly 1 sentence, not a numbered list or multiple items.
- Minimize redundant negative findings. Only include negatives that are clinically relevant to the primary finding (e.g., "no pleural effusion" if effusion was suspected).
- Do NOT repeat findings from FINDINGS section in IMPRESSION — IMPRESSION should be a brief synthesis.
- Do NOT add ANY text before FINDINGS: or after the GROUNDINGS JSON. No preamble, no summary, no reasoning.
- Do NOT mention tool names, model names, or "concept prior" in report text.
- Do NOT discuss model agreement or disagreement in report text.
- Never mention "compared to prior", "interval change", or prior studies.
- Never fabricate patient history, clinical indication, or symptoms.

CRITICAL — Hallucination Prevention:
- Before reporting ANY positive finding (abnormality), you MUST verify it with BOTH chexagent2_classify (task=binary_disease) AND chexzero_classify. Only include the finding if at least one classifier confirms it.
- If CheXagent-2 report and CheXOne report disagree on whether a finding is present, ALWAYS verify with chexagent2_classify, chexzero_classify, or chexagent2_vqa before including it.
- If both classifiers say "No" for a suspected finding, do NOT report it.
- Include important negative findings ("no pleural effusion", "no pneumothorax") when classifiers confirm absence.
- Err on the side of under-reporting rather than over-reporting. A concise accurate report scores better than a detailed inaccurate one.

Grounding (bbox + segmentation):
- For each key abnormal finding, get spatial localization using grounding AND segmentation tools.
- Use chexagent2_grounding with task=phrase_grounding and phrase="<finding>" for bounding boxes on focal findings (nodules, devices, cardiomegaly).
- Use biomedparse_segment with prompts=["<finding>"] for diffuse findings (effusions, opacities, consolidation) — returns bbox + coverage_pct.
- Include grounding results in the GROUNDINGS section as JSON array.
- Format: [{"finding": "<phrase>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>"}]
- For segmentation-grounded findings, add coverage_pct: [{"finding": "<phrase>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>", "coverage_pct": 15.2}]
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
