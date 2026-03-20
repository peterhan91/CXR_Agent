# CXR Agent — Feature Testing Protocol

Test 5 new capabilities one at a time against the `config_initial.yaml` baseline. Keep only features that improve **both** RadGraph F1 **and** 1/RadCliQ-v1. Combine winners into a final config.

## Strategy

**Independent-first, then combine.** Each feature tested in isolation on the same ~20-study stratified eval set. Features that pass the dual-metric gate are merged into `config_combined.yaml` and re-evaluated.

**Implementation order: 5 → 1 → 4 → 2 → 3 → combine.** Feature 5 goes first (zero server changes, pure agent modification). Features 2 and 3 depend on Feature 5's `_build_initial_message()` change (which makes `prior_report` and `prior_image_path` visible in initial mode).

## Features

Features 4 and 5 correspond to the reporting input diagram in `figures/reporting.png`: multiview images (frontal + lateral) and clinical context (demographics, indication, comparison).

| # | Feature | What changes | Servers needed |
|---|---------|-------------|----------------|
| 5 | Clinical metadata | Pass age/sex/indication/comparison to agent in initial mode | None new |
| 1 | MedGemma grounding | New `/ground` endpoint + `medgemma_grounding` tool | MedGemma (8010) |
| 4 | Multi-view input | Lateral CXR passed to report tools alongside frontal | Existing servers |
| 2 | MedGemma longitudinal | New `/longitudinal` endpoint + `medgemma_longitudinal` tool | MedGemma (8010) |
| 3 | CheXagent-2 temporal | New `/temporal` endpoint + `chexagent2_temporal` tool | CheXagent-2 (8001) |

---

## Step 0: Establish Baseline

### 0a. Sample stratified eval set

Sample ~20 studies (5 per dataset), stratified so every feature has signal. The eval set must satisfy:

| Requirement | Min studies | Source datasets |
|---|---|---|
| FINDINGS + IMPRESSION GT | All 20 | All 4 datasets |
| Clinical metadata (age/sex/indication) | >=15 | MIMIC-CXR, ReXGradient, CheXpert-Plus |
| Follow-up with prior image + report | >=5 | MIMIC-CXR (enriched), ReXGradient |
| Lateral view available | >=5 | MIMIC-CXR, IU-Xray, ReXGradient |

**Metadata availability per dataset** (from `prepare_eval_datasets.py`):

| Dataset | Age | Sex | Indication | Comparison | Prior study | Lateral |
|---------|-----|-----|-----------|------------|-------------|---------|
| MIMIC-CXR | Yes (MIMIC-IV `patients.csv`: `anchor_age + study_year - anchor_year`) | Yes (MIMIC-IV `patients.csv`: `gender`) | Via admission HPI (enriched `admission_info.patient_history`) | Report COMPARISON section | Yes (enriched, temporal linking) | Yes |
| CheXpert-Plus | Yes (CSV `age`) | Yes (CSV `sex`) | No | CSV `section_comparison` | No | No |
| ReXGradient | Yes (CSV `PatientAge`) | Yes (CSV `PatientSex`) | Yes (CSV `Indication`) | CSV `Comparison` | Yes (temporal linking) | Yes |
| IU-Xray | No | No | Partial (ReXrank `context`) | CSV comparison | No | Yes |

**Note on MIMIC-CXR metadata**: Age and sex require MIMIC-IV tables (`hosp/patients.csv`). Indication requires the enriched pipeline (`prepare_mimic_studies.py` → `admission_info`). The sampling script must check whether enriched data exists for candidate MIMIC-CXR studies to ensure metadata is actually available.

Write `scripts/sample_feature_eval.py`:

```python
# scripts/sample_feature_eval.py
#
# For each dataset, prefer studies that have:
# 1. Both FINDINGS and IMPRESSION in GT (required for all)
# 2. At least some with prior_study (image_path + report) — for Features 2, 3
# 3. At least some with lateral_image_path — for Feature 4
# 4. At least some with metadata (age, sex, indication) — for Feature 5
#
# MIMIC-CXR metadata: entry["metadata"] has subject_id but age/sex require
#   MIMIC-IV patients.csv join. Either:
#   (a) Use enriched data (admission_info.demographics.age/gender), or
#   (b) Join patients.csv at sampling time using subject_id + study date
#
# ReXGradient metadata: entry["metadata"] has age, sex, indication directly
# CheXpert-Plus metadata: entry["metadata"] has age, sex
# IU-Xray: no age/sex, partial indication via entry["metadata"]["indication"]
#
# Output: data/eval/feature_test/all_20.json
```

### 0b. Run baseline and score

```bash
# Run agent with initial config (no new features)
conda run -n cxr_agent python scripts/eval_mimic.py --mode agent \
  --input data/eval/feature_test/all_20.json \
  --output results/eval_features/baseline/ \
  --config configs/config_initial.yaml

# Export CSVs
conda run -n cxr_agent python scripts/eval_mimic.py --mode score \
  --output results/eval_features/baseline/

# Score with all 7 ReXrank metrics
bash scripts/score_rexrank.sh results/eval_features/baseline/
```

Record baseline RadGraph F1 and RadCliQ-v1. These are the numbers to beat.

### 0c. Quick review (3-5 CXRs) — applies to baseline AND each feature

For the quick review gate before full eval, do both:

**1. ReXrank scoring on the quick subset** — run `score_rexrank.sh` on just the 3-5 studies to get early metric signal. This catches obvious regressions before spending time on 20 studies.

**2. Claude-as-radiologist review** — Claude Code reads the agent's predicted report and ground truth, then produces a structured assessment:

```
For each study:
  1. Read predicted report + ground truth report
  2. Compare for: missing findings, hallucinated findings, style, clinical accuracy
  3. Output:
     {
       study_id: "...",
       score: 1-5,           # 1=unusable, 3=acceptable, 5=radiologist-quality
       missing_findings: [],  # findings in GT but not in prediction
       hallucinations: [],    # findings in prediction but not in GT
       style_notes: "...",    # brevity, format, hedging
       recommendation: "go" | "no-go"
     }
  4. Aggregate: go/no-go for the feature based on majority
```

This replaces manual radiologist review for the quick gate. If either the metrics or the qualitative review show clear regression, skip full eval.

---

## Feature 5: Clinical Metadata

**Goal**: Pass age, sex, indication, comparison text to the agent in initial mode.

**Why first**: Zero server/tool changes. Pure agent-level modification.

### What needs to change

**`agent/react_agent.py`** — `_build_initial_message()` (line 595-604)

Currently in initial mode, all context is discarded. Change to append `clinical_context` and `prior_report`/`prior_image_path` when provided. (This also unblocks Features 2 and 3 which need `prior_image_path` visible in initial mode.)

**`scripts/eval_mimic.py`** — `run_agent_eval()` (around line 670-684)

Currently metadata is only sourced from enriched data (requires `--use_clinical_context` + `--enriched_json`). Add a third source: entry-level metadata from `prepare_eval_datasets.py` output.

The eval JSON entries have these metadata fields (all under `entry["metadata"]`):

| Field path | MIMIC-CXR | CheXpert-Plus | ReXGradient | IU-Xray |
|---|---|---|---|---|
| `metadata.age` | **Missing** — see below | Yes (string) | Yes (string) | No |
| `metadata.sex` | **Missing** — see below | Yes (string) | Yes (string) | No |
| `metadata.indication` | No (see below) | No | Yes (string) | Yes (from ReXrank `context`) |
| `metadata.comparison` | Yes (report section) | Yes (CSV) | Yes (CSV) | Yes (CSV) |
| `metadata.subject_id` | Yes | No | No | No |
| `metadata.admission_info` | Yes if enriched | No | No | No |

**MIMIC-CXR age/sex**: Not stored directly in `metadata`. Two paths to get them:
- **(a)** Enriched pipeline: `metadata.admission_info.demographics.age` (int) and `.gender` ("M"/"F") — requires `prepare_mimic_studies.py` to have run
- **(b)** Lightweight join at sampling time: read MIMIC-IV `hosp/patients.csv`, compute `age = anchor_age + (study_year - anchor_year)`, get `gender`. Store in `metadata.age` and `metadata.sex` in the sampled JSON so it's self-contained.

**MIMIC-CXR indication**: Only available via enriched `metadata.admission_info.patient_history` (full HPI text from discharge note). No concise indication field exists.

Option (b) for age/sex is simpler — do it in `sample_feature_eval.py`. For indication, use `admission_info.patient_history` if enriched data exists, otherwise omit.

**`configs/config_feat5_metadata.yaml`** — Copy of `config_initial.yaml`. No config flag needed — the `_build_initial_message()` change just stops discarding context that's already being passed.

### Eval

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode agent \
  --input data/eval/feature_test/all_20.json \
  --output results/eval_features/feat5_metadata/ \
  --config configs/config_feat5_metadata.yaml

conda run -n cxr_agent python scripts/eval_mimic.py --mode score \
  --output results/eval_features/feat5_metadata/

bash scripts/score_rexrank.sh results/eval_features/feat5_metadata/
```

Compare RadGraph F1 and RadCliQ-v1 vs baseline.

---

## Feature 1: MedGemma Grounding

**Goal**: Test if MedGemma-1.5 bounding boxes improve spatial localization vs CheXagent-2 grounding alone. Two variants:
- **1a**: Add `medgemma_grounding` alongside `chexagent2_grounding`
- **1b**: Replace `chexagent2_grounding` with `medgemma_grounding`

### What needs to change

**`servers/medgemma_server.py`** — Add `/ground` endpoint

- New `GroundingRequest`: `image_path`, `phrase`, `max_new_tokens`
- New `GroundingResponse`: `result`, `boxes` (list of `{x_min, y_min, x_max, y_max}` normalized 0-1), `generation_time_ms`
- Pad image to square (reuse `_pad_to_square` pattern from chexagent2_server)
- Prompt: `"Where is {phrase} in this chest X-ray?"` with system prompt for `[y0, x0, y1, x1]` output in `[0, 1000]` range
- Parse bounding boxes from model output, convert `[y0, x0, y1, x1]` to `{x_min, y_min, x_max, y_max}`, correct for padding

**`tools/medgemma.py`** — Add `MedGemmaGroundingTool`

- `name`: `"medgemma_grounding"`
- Description: `[GROUNDING]` with WHEN TO USE guidance, similar to `CheXagent2GroundingTool`
- `input_schema`: `image_path` (required), `phrase` (required)
- `run()`: POST to `{endpoint}/ground`, format output like `CheXagent2GroundingTool`

**`agent/initial_mode.py`** — Add `"medgemma_grounding"` to `INITIAL_TOOL_DESCRIPTIONS`

**`tools/__init__.py`** — Export `MedGemmaGroundingTool`

**`scripts/run_agent.py` + `scripts/eval_mimic.py`** — Add `"medgemma_grounding": MedGemmaGroundingTool` to tool registry

**Configs**:
- `configs/config_feat1a_grounding_add.yaml` — baseline + `medgemma_grounding: enabled: true`
- `configs/config_feat1b_grounding_replace.yaml` — baseline with `chexagent2_grounding: enabled: false`, `medgemma_grounding: enabled: true`

### Eval

Run both 1a and 1b. Compare each to baseline. Keep the better option (or neither if both degrade).

---

## Feature 4: Multi-view Input (Frontal + Lateral)

**Goal**: Pass lateral CXR alongside frontal to report generation tools for more comprehensive reports.

### What needs to change

**`agent/react_agent.py`**
- Add `lateral_image_path: str = ""` parameter to `run()` method
- In `_build_initial_message()`: when `lateral_image_path` is provided, append `"\n\nLateral view image path: {lateral_image_path}"` to text

**Report tool schemas** — Add optional `lateral_image_path` to `input_schema` for:
- `tools/chexagent2.py`: `CheXagent2ReportTool`, `CheXagent2SRRGTool`
- `tools/chexone.py`: `CheXOneReportTool`
- `tools/medgemma.py`: `MedGemmaReportTool`

**Server endpoints** — Accept optional second image:
- `servers/chexagent2_server.py`: Add `lateral_image_path: Optional[str] = None` to `ReportRequest`/`SRRGRequest`. Build `image_paths = [req.image_path] + ([req.lateral_image_path] if req.lateral_image_path else [])`. Already supports multi-image via `generate(model, tokenizer, image_paths: list, ...)`.
- `servers/chexone_server.py`: Add second `{"type": "image", "image": _load_cxr(req.lateral_image_path)}` to messages content when provided.
- `servers/medgemma_server.py`: Add second image to messages content when provided.

**`scripts/eval_mimic.py`** — Pass `entry.get("lateral_image_path", "")` to `agent.run()`.

**Config**: `configs/config_feat4_multiview.yaml` — Same tools as baseline. Feature is data-driven (activates when lateral exists in eval data).

### Eval

The stratified eval set includes >=5 studies with lateral views (from MIMIC-CXR, IU-Xray, ReXGradient per `prepare_eval_datasets.py`). Score overall 20 studies + report lateral-subset metrics separately.

---

## Feature 2: MedGemma Longitudinal Comparison

**Goal**: Better follow-up reporting (progression/regression/stable) using MedGemma's multi-image comparison.

**Depends on**: Feature 5's `_build_initial_message()` change (which makes `prior_image_path` visible in initial mode).

### What needs to change

**`servers/medgemma_server.py`** — Add `/longitudinal` endpoint

- New `LongitudinalRequest`: `current_image_path`, `prior_image_path`, `max_new_tokens`
- Load both images, build messages with prior (first) and current (second)
- Prompt: `"Compare these two chest X-rays. The first is the prior study and the second is the current study. Describe interval changes."`

**`tools/medgemma.py`** — Add `MedGemmaLongitudinalTool`

- `name`: `"medgemma_longitudinal"`
- Description: `[LONGITUDINAL]` with WHEN TO USE: only when `prior_image_path` is available
- `input_schema`: `current_image_path` (required), `prior_image_path` (required)

**`agent/initial_mode.py`** — Add to `INITIAL_TOOL_DESCRIPTIONS`

**Registry** — Add to `_build_tools()` in both `run_agent.py` and `eval_mimic.py`, and to `tools/__init__.py`

**Config**: `configs/config_feat2_longitudinal_mg.yaml` — baseline + `medgemma_longitudinal: enabled: true`

### Eval

Eval set has >=5 follow-up studies with prior image + report (from MIMIC-CXR enriched, ReXGradient). Score overall + follow-up subset.

---

## Feature 3: CheXagent-2 Temporal Comparison

**Goal**: Leverage CheXagent-2's trained temporal classification (improved/stable/worsened) for follow-up reporting.

**Depends on**: Feature 5's `_build_initial_message()` change.

### What needs to change

**`servers/chexagent2_server.py`** — Add `/temporal` endpoint

- New `TemporalRequest`: `current_image_path`, `prior_image_path`, `disease_name: Optional[str] = None`, `max_new_tokens`
- If `disease_name` provided: use CheXagent-2's `temporal_image_classification` task — `"Given two images of a patient, where the first is taken before the second, determine if {disease_name} has improved, worsened, or stabilized."`
- If no `disease_name`: open-ended comparison via `findings_generation` with 2 images — `"Compare the findings between these two chest X-rays taken at different times."`
- Image order: `[prior_image_path, current_image_path]` (chronological, as CheXagent-2 expects)

**`tools/chexagent2.py`** — Add `CheXagent2TemporalTool`

- `name`: `"chexagent2_temporal"`
- Description: `[TEMPORAL]` with WHEN TO USE: only when prior study image is available
- `input_schema`: `current_image_path` (required), `prior_image_path` (required), `disease_name` (optional)

**`agent/initial_mode.py`** — Add to `INITIAL_TOOL_DESCRIPTIONS`

**Registry** — Add to `_build_tools()` and `tools/__init__.py`

**Config**: `configs/config_feat3_temporal_ca2.yaml` — baseline + `chexagent2_temporal: enabled: true`

### Eval

Same follow-up subset as Feature 2. Score overall + follow-up subset.

---

## Phase 2: Combine Winners

After all 5 features tested independently:

1. Identify features where BOTH RadGraph F1 AND 1/RadCliQ-v1 improved vs baseline
2. Create `configs/config_combined.yaml` with all winners enabled
3. Run on same 20-study eval set, score with `score_rexrank.sh`
4. If combined still beats baseline → new production config
5. If combined regresses → ablate (remove features one at a time) to find best subset

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode agent \
  --input data/eval/feature_test/all_20.json \
  --output results/eval_features/combined/ \
  --config configs/config_combined.yaml

conda run -n cxr_agent python scripts/eval_mimic.py --mode score \
  --output results/eval_features/combined/

bash scripts/score_rexrank.sh results/eval_features/combined/
```

---

## Per-Feature Protocol Summary

For each feature:

1. **Quick review (3-5 CXRs)**: Run agent on 3-5 studies. Score with `score_rexrank.sh`. Claude Code acts as radiologist — compares report to GT, checks for hallucinations/missing findings. Go/no-go before full eval.
2. **Full eval (~20 CXRs)**: Run on full stratified eval set. Score with `score_rexrank.sh`.
3. **Decision**: If BOTH RadGraph F1 AND 1/RadCliQ-v1 improve vs baseline → keep. Otherwise → revert.

---

## Critical Files Summary

| File | Features Affected | Changes |
|------|------------------|---------|
| `agent/react_agent.py` | 5, 4, 2, 3 | `_build_initial_message()`: stop discarding context in initial mode + add `lateral_image_path` param; `run()`: add `lateral_image_path` param + pass through to `_build_initial_message()` (call at line 263) |
| `agent/initial_mode.py` | 1, 2, 3 | Add tool descriptions for new tools |
| `servers/medgemma_server.py` | 1, 2, 4 | Add `/ground`, `/longitudinal` endpoints; modify `/generate_report` for lateral |
| `servers/chexagent2_server.py` | 3, 4 | Add `/temporal` endpoint; modify `/generate_report`, `/generate_srrg` for lateral |
| `servers/chexone_server.py` | 4 | Modify `/generate_report` for lateral image |
| `tools/medgemma.py` | 1, 2 | Add `MedGemmaGroundingTool`, `MedGemmaLongitudinalTool` |
| `tools/chexagent2.py` | 3, 4 | Add `CheXagent2TemporalTool`; add `lateral_image_path` to report/srrg schemas |
| `tools/chexone.py` | 4 | Add `lateral_image_path` to schema |
| `tools/__init__.py` | 1, 2, 3 | Export new tool classes |
| `scripts/eval_mimic.py` | All | `_build_tools()` registry + `run_agent_eval()` metadata/lateral passthrough |
| `scripts/run_agent.py` | All | Mirror registry changes from eval_mimic.py |
| `scripts/sample_feature_eval.py` | 0 | New: stratified sampling with metadata join |
| `configs/config_feat*.yaml` | All | One config per feature test |

---

## Dependencies Between Features

```
Feature 5 (_build_initial_message change)
  ├── Feature 2 (needs prior_image_path visible in initial mode)
  └── Feature 3 (needs prior_image_path visible in initial mode)

Feature 1 (fully independent)
Feature 4 (independent, but shares server changes with 1-3)
```

---

## Setup Reference

| Port | Server | GPU | Tools |
|------|--------|-----|-------|
| 8001 | CheXagent-2 | 0 | `chexagent2_report`, `chexagent2_srrg_report`, `chexagent2_grounding`, `chexagent2_classify`, `chexagent2_vqa`, `chexagent2_temporal` (new) |
| 8002 | CheXOne | 1 | `chexone_report` |
| 8005 | BiomedParse | 1 | `biomedparse_segment` |
| 8007 | FactCheXcker | 2 | `factchexcker_verify` |
| 8010 | MedGemma | 2 | `medgemma_vqa`, `medgemma_report`, `medgemma_grounding` (new), `medgemma_longitudinal` (new) |

| Conda env | Purpose |
|-----------|---------|
| `cxr_agent` | Main agent + all servers except CheXagent-2 |
| `cxr_chexagent2` | CheXagent-2 server only (transformers==4.40.0) |
| `radgraph` | CXR-Report-Metric scoring (BLEU, BERT, Semb, RadGraph, RadCliQ) |
| `green_score` | RaTEScore + GREEN scoring |

## Verification

For each feature:
1. Start relevant servers
2. Quick test: `python scripts/run_agent.py --image <test_cxr> --config configs/config_feat*.yaml`
3. Quick review (3-5 studies): run + score + Claude-as-radiologist review
4. Full eval: `python scripts/eval_mimic.py --mode agent --config configs/config_feat*.yaml ...`
5. Score: `bash scripts/score_rexrank.sh <output_dir>/`
6. Compare: `python scripts/eval_mimic.py --mode compare --output <output_dir>/`
