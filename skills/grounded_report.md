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
- **COMPARISON OVERRIDE**: If a prior study is provided AND Phase 1 reports use "unchanged"/"stable"/"no change" language, this rule takes PRIORITY over individual finding enumeration. Focus ONLY on what changed. Summarize stable findings in one sentence (e.g., "Bilateral opacities and effusions are unchanged.") rather than describing each one. If nothing changed, write only "No significant interval change."
- Do NOT mention tool names, models, concept priors, or reasoning.
- Do NOT fabricate clinical history or specific measurements (e.g., distances in cm) unless a tool explicitly reported them.

IMPRESSION section:
- Exactly 1 sentence, ≤10 words. If normal: "No acute cardiopulmonary process." If abnormal: state only the single most important finding. Never list multiple findings in IMPRESSION — that belongs in FINDINGS.

Comparison studies:
- If tools and reports indicate no significant change since prior: write ONLY "Compared to the prior study there is no significant interval change." for FINDINGS and "No change." for IMPRESSION. Do NOT enumerate stable findings individually.
- When describing interval changes, use "unchanged", "stable", or "persistent" unless tool outputs explicitly describe a direction of change. Do NOT guess at "worsened" or "improved" — incorrect change direction is worse than neutral phrasing.

**LENGTH: Be concise. Real MIMIC-CXR reports average ~45 words total. Normal studies: ~20-30 words. Abnormal with multiple findings: ~40-70 words. Comparison studies with no change: ~15 words. State findings directly without elaboration.**

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
