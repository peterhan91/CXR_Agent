---
name: grounded_report
version: "4.1"
description: CXR report generation with grounding, optimized for RadCliQ-v1/RadGraph-F1/BLEU-2
---

# Workflow: Generate Grounded CXR Report

Execute these phases IN ORDER. Do not skip any mandatory step.

## Phase 1: Collect Reports (3 calls — MANDATORY)
Call all three report generators. Each provides a different clinical perspective.
1. `chexagent2_report` — free-text report
2. `chexone_report` — second-opinion report
3. `chexagent2_srrg_report` — structured region-by-region report

## Phase 2: Screen All Pathologies (2 calls — MANDATORY)
4. `chexzero_classify` — CheXzero zero-shot 14-label screening. Record all labels marked PRESENT.
5. `cxr_foundation_classify` — CXR Foundation zero-shot 14-label screening. Record all labels marked PRESENT.

## Phase 3: Confirm Suspected Findings (1-3 calls)
For each finding flagged positive by Phase 1 reports OR Phase 2 screening:
6. `chexagent2_classify` with `task="binary_disease"`, `disease_name="<finding>"` — binary confirmation.
7. If Phase 1 reports disagree on a finding, call `chexagent2_vqa` with a direct yes/no question (e.g., "Is there a pleural effusion?").

Decision rule:
- INCLUDE a finding if at least 2 of 3 classifiers agree (chexzero, cxr_foundation, chexagent2_classify).
- EXCLUDE if 2+ classifiers say no, even if a report mentions it.

## Phase 4: Ground Confirmed Findings (1-3 calls)
For each confirmed abnormal finding:
8. `chexagent2_grounding` with `task="phrase_grounding"`, `phrase="<finding>"` — bounding box for focal findings (cardiomegaly, nodule, device).
9. `biomedparse_segment` with `prompts=["<finding>"]` — segmentation for diffuse findings (effusion, opacity, consolidation, atelectasis). Use coverage_pct to validate extent.

## Phase 5: Verify (1 call — MANDATORY)
10. `factchexcker_verify` with your draft report text. If it flags errors, fix them.

## Phase 6: Write Final Report

### CRITICAL — Match the Ground Truth Report Style

Write reports that match standard radiology dictation style. The Phase 1 reports provide a style reference for this specific study — mirror their tone, phrasing, and level of detail.

### Writing Rules

FINDINGS section:
- Write plain prose. No bullets, no headers, no markdown.
- Do NOT start with technique ("PA view of the chest", "AP radiograph") — jump straight to findings.
- Prefer "There is..." / "There is no..." phrasing for stating/negating findings (e.g., "There is no pleural effusion.", "There is no pneumothorax.").
- Use standard radiology phrases: "is seen", "are noted", "is normal", "are unremarkable".
- Combine negatives: "There is no focal consolidation, pleural effusion or pneumothorax."
- Include ALL findings from tool outputs — do not omit subtle ones (atelectasis, old fractures, scarring, calcifications).
- If prior study provided: focus on what CHANGED. Use "unchanged", "improved", "worsened".
- Do NOT mention tool names, models, concept priors, or reasoning.
- Do NOT fabricate clinical history or specific measurements (e.g., distances in cm) unless a tool explicitly reported them.

IMPRESSION section:
- 1-2 sentences summarizing the key findings. If normal: "No acute cardiopulmonary process."

Comparison studies:
- When describing interval changes, use "unchanged", "stable", or "persistent" unless tool outputs explicitly describe a direction of change. Do NOT guess at "worsened" or "improved" — incorrect change direction is worse than neutral phrasing.

### Output Format
```
FINDINGS:
<plain text paragraph>

IMPRESSION:
<1-2 sentence summary>

GROUNDINGS:
[{"finding": "...", "bbox": [x_min, y_min, x_max, y_max], "tool": "..."}]
```

GROUNDINGS: JSON array with one entry per grounded finding. Include `coverage_pct` from biomedparse when available. Only include groundings for findings that appear in the final FINDINGS/IMPRESSION text — if a finding was grounded in Phase 4 but later excluded from the report, do NOT include its grounding.
