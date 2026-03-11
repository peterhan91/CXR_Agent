---
name: grounded_report
version: "1.0"
description: Grounded CXR report generation with spatial verification
---

# Grounded Report Generation Workflow

## Phase 1: Gather Findings (2-3 tool calls)
1. Call `chexagent2_report` to get a free-text report.
2. Call `chexone_report` for a second opinion.
3. Call `chexagent2_srrg` for structured region-by-region findings.

## Phase 2: Classify and Confirm (1-2 tool calls)
4. Call `chexagent2_classify` with `task=binary_disease` for key suspected pathologies to confirm presence/absence.
5. If conflicting findings, call `chexagent2_vqa` with a targeted question to resolve.

## Phase 3: Ground Key Findings (1-3 tool calls)
6. For each key abnormal finding, call `chexagent2_grounding` to get its bounding box:
   - Use `task=phrase_grounding` with `phrase="<finding text>"` for specific findings.
   - Use `task=abnormality` with `disease_name="<pathology>"` for disease localization.
7. Record the bounding box coordinates for each grounded finding.

## Phase 4: Write Report
8. Synthesize a concise report in MIMIC-CXR style (see format rules below).
9. Append a GROUNDINGS section mapping findings to bounding boxes.

## Report Format Rules
- Write plain text. NO markdown (no ##, **, -, bullets).
- FINDINGS section: one concise paragraph, 60-100 words.
- IMPRESSION section: numbered key findings, 1-4 items.
- Total report (FINDINGS + IMPRESSION): 80-150 words.
- Use standard radiology phrasing: "There is...", "No evidence of...", "The cardiac silhouette is...".
- Do NOT mention tool names, model names, concept priors, or internal reasoning.
- Do NOT discuss model agreement/disagreement in the report text.
- Do NOT use hedging language like "possibly", "may be" unless clinically appropriate.
- Do NOT reference prior studies or interval change unless explicitly provided.

## Grounding Format
After IMPRESSION, include a GROUNDINGS section as a JSON array:
```
GROUNDINGS:
[{"finding": "<phrase from report>", "bbox": [x_min, y_min, x_max, y_max], "tool": "<tool_name>"}]
```
- bbox uses normalized coordinates [0,1].
- Only include findings that were successfully grounded by a tool.
- If a grounding tool returned no result for a finding, omit it from GROUNDINGS.
