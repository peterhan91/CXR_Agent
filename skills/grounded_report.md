---
name: grounded_report
version: "3.0"
description: Grounded CXR report generation using ALL 9 tools for maximum accuracy
---

# Grounded Report Generation Workflow

You have 9 tools. Use ALL of them — more tools = more accurate reports = higher scores.
Do NOT use any MedVersa tools (medversa_*) or MedSAM tools — they are disabled.

## Phase 1: Gather Findings (3 tool calls — MANDATORY)
1. Call `chexagent2_report` — free-text report.
2. Call `chexone_report` — second opinion from a different model.
3. Call `chexagent2_srrg_report` — structured region-by-region findings (Lungs, Pleura, Cardiovascular, Other).

## Phase 2: Screen and Classify (2-4 tool calls — MANDATORY)
4. Call `chexzero_classify` — zero-shot screening of all 14 CheXpert pathologies at once. P>0.5 = positive. Best for cardiomegaly, edema, pleural effusion.
5. Call `chexagent2_classify` with `task=binary_disease` for EACH finding flagged by Phase 1 reports OR chexzero (e.g., disease_name="cardiomegaly").
6. If CheXagent-2 and CheXOne disagree, resolve with `chexagent2_vqa` (e.g., "Is there evidence of pleural effusion in this chest X-ray?").
7. Only include findings confirmed by at least one classifier. Drop anything both classifiers reject.

## Phase 3: Ground with Bounding Boxes (1-3 tool calls)
For each CONFIRMED abnormal finding:
8. Call `chexagent2_grounding` with `task=phrase_grounding` and `phrase="<finding>"` (e.g., "cardiomegaly", "right pleural effusion", "left basilar opacity").

## Phase 4: Segment Diffuse Findings (1-2 tool calls)
For diffuse/regional findings (effusions, opacities, consolidation, atelectasis):
9. Call `biomedparse_segment` with `prompts=["<finding>"]` to get segmentation coverage and bbox.
   - Good prompts: "lung opacity", "left lung", "right lung", "pleural effusion", "viral pneumonia".
   - Returns: coverage_pct (extent of finding), bbox (bounding region), mask_shape.
10. Use segmentation coverage to VALIDATE findings — 0% coverage = likely absent; >40% = diffuse.

## Phase 5: Verify Report (1 tool call — MANDATORY)
11. ALWAYS call `factchexcker_verify` with your draft report text before finalizing.
    - It checks for hallucinations, measurement errors, and factual inconsistencies.
    - If changes_made is true, incorporate corrections.

## Phase 6: Write Final Report
12. Synthesize a concise report in MIMIC-CXR style following all format rules.
13. For GROUNDINGS, include BOTH bbox and segmentation data:
    - Use bbox from chexagent2_grounding for focal findings.
    - Use bbox from biomedparse_segment for diffuse findings.
    - Include coverage_pct when available.

## Tool Call Target: 8-12 calls per study
Minimum workflow: 3 reports + 1 chexzero screen + 1-2 classify + 1-2 grounding + 1 factcheck = 8-10 calls.
Add biomedparse_segment and chexagent2_vqa as needed for 10-12 calls.

## Report Format Rules
- Write plain text. NO markdown (no ##, **, -, ---, bullets, headers).
- FINDINGS: one concise paragraph, 35-55 words.
- IMPRESSION: exactly 1 sentence synthesizing the key finding.
- Total (FINDINGS + IMPRESSION): 50-75 words combined.
- Use standard radiology phrasing: "There is...", "No evidence of...", "The cardiac silhouette is...".
- Do NOT mention tool names, model names, concept priors, or reasoning in report text.
- Do NOT discuss model agreement/disagreement.
- ALWAYS verify positive findings with classification before including.
- Include relevant negative findings when confirmed.
- Do NOT reference prior studies or interval change.

## Grounding Format
```
GROUNDINGS:
[
  {"finding": "<phrase>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>"},
  {"finding": "<phrase>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>", "coverage_pct": 15.2}
]
```
- bbox: normalized [0,1] coordinates.
- Include coverage_pct from segmentation tools when available.
- Only include findings successfully grounded by a tool.
- Prefer chexagent2_grounding bbox for focal findings; biomedparse_segment bbox for diffuse findings.
