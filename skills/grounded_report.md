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

**Example 1 (normal):**
> FINDINGS:
> PA and lateral views of the chest. There is no focal consolidation, pleural effusion or pneumothorax. The cardiomediastinal and hilar contours are normal.
>
> IMPRESSION:
> No acute cardiopulmonary process.

**Example 2 (normal):**
> FINDINGS:
> Lungs are clear. There is no effusion or consolidation. Cardiomediastinal silhouette is normal. No acute osseous abnormalities identified.
>
> IMPRESSION:
> No acute cardiopulmonary process.

**Example 3 (abnormal):**
> FINDINGS:
> Cardiac silhouette size is normal. Mediastinal and hilar contours are unremarkable. Ill-defined patchy opacities are noted in the left lung base, concerning for pneumonia. Blunting of the costophrenic angles bilaterally suggests trace bilateral pleural effusions.
>
> IMPRESSION:
> Patchy ill-defined left basilar opacity concerning for pneumonia. Small bilateral pleural effusions.

### Writing Rules

FINDINGS section:
- Write 2-5 sentences of plain prose. No bullets, no headers, no markdown.
- Start with anatomy/cardiac assessment, then lung findings, then other.
- Use passive/descriptive voice: "is seen", "are noted", "is present", "is normal", "are unremarkable".
- Standard phrases to prefer:
  - Normal heart: "Cardiac silhouette size is normal" or "Cardiomediastinal silhouette is normal"
  - Normal lungs: "Lungs are clear" or "No focal consolidation"
  - Normal pleura: "No pleural effusion" or "No pneumothorax"
  - Combined normal: "No focal consolidation, pleural effusion or pneumothorax"
  - Cardiomegaly: "The cardiac silhouette is enlarged" or "Cardiomegaly"
  - Effusion: "Small bilateral pleural effusions" or "Blunting of the costophrenic angles"
  - Opacity: "Patchy opacity in the [location]" or "Ill-defined opacity"
  - Edema: "Pulmonary vascular congestion" or "Interstitial edema"
  - Atelectasis: "Bibasilar atelectasis" or "Patchy atelectasis at the lung bases"
- Include relevant negatives (e.g., "No pneumothorax" when effusion is present).
- Do NOT mention tool names, model names, concept priors, or your reasoning process.
- Do NOT reference prior studies, interval change, or comparison.
- Do NOT fabricate clinical history or indications.

IMPRESSION section:
- Exactly 1-2 sentences summarizing the key clinical message.
- If normal: "No acute cardiopulmonary process." (use this exact phrase when appropriate)
- If abnormal: state the primary finding and any important secondary findings.
- Do NOT repeat all FINDINGS — synthesize.

Length target: FINDINGS 30-80 words, IMPRESSION 5-20 words. Total 40-90 words.

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
