---
name: grounded_report
version: "4.0"
description: MIMIC-style CXR report generation optimized for RadCliQ-v1/RadGraph-F1/BLEU-2
---

# Workflow: Generate MIMIC-CXR Report

Execute these phases IN ORDER. Do not skip any mandatory step.

## Phase 1: Collect Reports (3 calls — MANDATORY)
Call all three report generators. Each provides a different clinical perspective.
1. `chexagent2_report` — free-text report
2. `chexone_report` — second-opinion report
3. `chexagent2_srrg_report` — structured region-by-region report

## Phase 2: Screen All Pathologies (2 calls — MANDATORY)
4. `chexzero_classify` — CheXzero zero-shot 14-label screening. Record all labels marked PRESENT.
5. `cxr_foundation_classify` — CXR Foundation zero-shot 14-label screening. Record all labels marked PRESENT.

## Phase 3: Confirm Suspected Findings (1-4 calls)
For each finding flagged positive by Phase 1 reports OR Phase 2 screening:
6. `chexagent2_classify` with `task="binary_disease"`, `disease_name="<finding>"` — binary confirmation.
7. If Phase 1 reports disagree on a finding, call `chexagent2_vqa` with a direct yes/no question (e.g., "Is there a pleural effusion?").
8. Use `medgemma_vqa` for a second-opinion VQA when classifiers are split (e.g., "Does this chest X-ray show cardiomegaly?").

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

### CRITICAL — Match MIMIC-CXR Report Style Exactly

Study these real MIMIC-CXR examples and match their tone, structure, and phrasing:

**Example 1 (normal, 23 words):**
> FINDINGS:
> The lungs are clear. The hilar and cardiomediastinal contours are normal. There is no pneumothorax. There is no pleural effusion.
>
> IMPRESSION:
> Normal chest.

**Example 2 (normal, 27 words):**
> FINDINGS:
> The lungs are clear. There is no focal consolidation, pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal. Pulmonary vascularity is normal.
>
> IMPRESSION:
> No acute cardiopulmonary process.

**Example 3 (comparison, stable, 15 words):**
> FINDINGS:
> Compared to the prior study there is no significant interval change.
>
> IMPRESSION:
> No change.

**Example 4 (abnormal, 35 words):**
> FINDINGS:
> Cardiac silhouette size is normal. Ill-defined patchy opacities are noted in the left lung base. Blunting of the costophrenic angles bilaterally suggests trace bilateral pleural effusions.
>
> IMPRESSION:
> Left basilar opacity concerning for pneumonia. Small bilateral pleural effusions.

### Writing Rules

FINDINGS section:
- Write 2-4 sentences of plain prose. No bullets, no headers, no markdown.
- Do NOT start with technique ("PA view of the chest", "AP radiograph") — jump straight to findings.
- Start with cardiac/mediastinal assessment, then lungs, then other.
- Prefer "There is..." / "There is no..." phrasing for stating/negating findings (e.g., "There is no pleural effusion.", "There is no pneumothorax."). This matches standard MIMIC-CXR dictation style.
- Use standard radiology phrases: "is seen", "are noted", "is normal", "are unremarkable".
- Combine negatives: "There is no focal consolidation, pleural effusion or pneumothorax."
- Include ALL findings from tool outputs — do not omit subtle ones (atelectasis, old fractures, scarring, calcifications).
- If prior study provided: focus on what CHANGED. Use "unchanged", "improved", "worsened". Do NOT re-describe unchanged devices or findings — a single sentence like "Support devices are unchanged in position." suffices. If nothing changed, the entire report can be as brief as "No significant interval change."
- Do NOT mention tool names, models, concept priors, or reasoning.
- Do NOT fabricate clinical history or specific measurements (e.g., distances in cm) unless a tool explicitly reported them.

IMPRESSION section:
- 1 sentence. If normal: "No acute cardiopulmonary process." If abnormal: state key finding(s).

**LENGTH: Be concise. Real MIMIC-CXR reports average ~45 words total. Normal studies: ~20-30 words. Abnormal with multiple findings: ~40-70 words. Avoid verbose descriptions — state findings directly without elaboration. IMPRESSION is always 1 sentence.**

### Output Format
```
FINDINGS:
<plain text paragraph>

IMPRESSION:
<1-2 sentence summary>

GROUNDINGS:
[{"finding": "...", "bbox": [x_min, y_min, x_max, y_max], "tool": "..."}]
```

GROUNDINGS: JSON array with one entry per grounded finding. Include `coverage_pct` from biomedparse when available. Only include findings that were successfully grounded by a tool.
