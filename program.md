# CXR Agent

Autonomous experimentation for CXR report generation. The agent reads this file, runs experiments, and iterates — no human in the loop.

## Goal

**Maximize all 7 ReXrank metrics on MIMIC-CXR test set.** Beat ALL baselines (Sonnet API, CheXOne standalone) on ALL metrics. Use every available tool to get there.

| Metric | Column | Range | Higher is Better | Measures |
|--------|--------|-------|-------------------|----------|
| BLEU-2 | bleu_score | [0, 1] | Yes | Bigram overlap (weights: 1/2, 1/2) via fast_bleu |
| BERTScore | bertscore | [-1, 1] | Yes | Contextual embedding F1 (distilroberta-base, baseline rescaling, no IDF) |
| SembScore | semb_score | [-1, 1] | Yes | Cosine similarity of CheXbert embeddings |
| RadGraph | radgraph_combined | [0, 1] | Yes | Mean of entity F1 and relation F1 from RadGraph (DyGIE++) |
| 1/RadCliQ-v1 | 1/RadCliQ-v1 | (0, +inf) | Yes | Inverse of RadCliQ-v1 composite (primary ranking metric) |
| RaTEScore | ratescore | [0, 1] | Yes | Factual/temporal consistency score |
| GREEN | green_score | [0, 1] | Yes | LLM-based clinical error analysis (StanfordAIMI/GREEN-radllama2-7b) |

**Primary optimization target**: 1/RadCliQ-v1 (composites BLEU-2 + BERTScore + SembScore + RadGraph). RaTEScore and GREEN are secondary but must not regress.

Per study, output: report text (FINDINGS + IMPRESSION), grounded findings (JSON with bboxes), two figures (`{id}_bbox.png`, `{id}_mask.png`).

## Setup

Before starting the experiment loop, complete these steps in order:

### Step 1: Verify tool servers (10 active)

Every tool server must respond. Send a health-check request to each:

```bash
# Quick health check — all should return 200
for port in 8001 8002 8005 8007 8008 8009; do
  echo -n "Port $port: "; curl -sf http://localhost:$port/health && echo "OK" || echo "FAIL"
done
```

| Tool | Server | What it does |
|------|--------|-------------|
| `chexagent2_report` | :8001 | Free-text report |
| `chexagent2_srrg_report` | :8001 | Structured report by anatomy |
| `chexagent2_grounding` | :8001 | Bbox per finding (`task=phrase_grounding`, `phrase="..."`) |
| `chexagent2_classify` | :8001 | Binary disease classification (`task=binary_disease`, `disease_name="..."`) |
| `chexagent2_vqa` | :8001 | Follow-up questions |
| `chexone_report` | :8002 | Second-opinion report (Qwen2.5-VL) |
| `chexzero_classify` | :8009 | CheXzero zero-shot 14-label screening (CLIP ViT-B/32 + logit_scale) |
| `cxr_foundation_classify` | :8008 | Google CXR Foundation zero-shot 14-label screening (ELIXR v2, CPU) |
| `biomedparse_segment` | :8005 | Text-prompted segmentation (`prompts=["left lung"]`; good for anatomy, not pathology) |
| `factchexcker_verify` | :8007 | Report hallucination checker |

Plus **CLEAR concept prior** — DINOv2+CLIP cosine similarity to MIMIC-CXR concepts, injected before tool calls.

**Disabled**: MedVersa (hallucinating), MedSAM (poor CXR masks), MedSAM3 (replaced).

### Step 2: Validate FactCheXcker actually catches errors

FactCheXcker must detect real inconsistencies — not just rubber-stamp every report as correct. Test it with a deliberately wrong report:

```bash
# Send a known-bad report for a normal CXR image
curl -X POST http://localhost:8007/verify \
  -H "Content-Type: application/json" \
  -d '{"image_path": "<any_val_cxr_image>", "report": "Large bilateral pleural effusions with complete left lung opacification. Massive cardiomegaly. Multiple bilateral pulmonary nodules suspicious for metastatic disease."}'
```

**Expected**: FactCheXcker should flag inconsistencies (e.g., "no evidence of pleural effusion"). If it returns "all findings verified" for an obviously wrong report on a normal CXR, the tool is broken — investigate before proceeding. Also test with a correct report for the same image to confirm it passes.

### Step 3: Verify ReXrank scoring services

All 7 metrics must be computable. Test with a small dummy run:

```bash
# Verify CXR-Report-Metric (5 core metrics: BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ-v1)
conda run -n radgraph python -c "from CXRMetric.run_eval import calc_metric; print('CXR-Report-Metric OK')"

# Verify RaTEScore
conda run -n green_score python -c "from RaTEScore import RaTEScore; print('RaTEScore OK')"

# Verify GREEN (loads 7B model — takes ~30s first time)
conda run -n green_score python -c "from green_score.green import GREEN; print('GREEN OK')"
```

If any fail, fix the conda environment before proceeding. Scoring with missing metrics is not acceptable.

### Step 4: Prepare enriched data (prior studies + clinical context)

The agent runs with `--use_prior` and `--use_clinical_context` enabled by default. These require enriched JSON files linking CXR studies to MIMIC-IV admissions and prior studies.

```bash
# Check if enriched data already exists
ls results/eval_enriched/val_studies_enriched.json results/eval_enriched/test_studies_enriched.json 2>/dev/null

# If missing, generate it (requires MIMIC-IV access):
python scripts/prepare_mimic_studies.py \
  --mimic_cxr_dir $MIMIC \
  --mimic_iv_dir /path/to/mimiciv/3.1 \
  --output results/eval_enriched/ \
  --splits validate,test
```

The enriched JSON provides per-study: prior CXR reports, prior image paths, admission info (HPI, chief complaint, demographics, ICD codes).

### Step 5: Prepare test sets and run baselines

```bash
MIMIC=/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0
CFG=configs/config_grounded.yaml
ENRICHED=results/eval_enriched/val_studies_enriched.json

# Prepare validation set (10 unique patients, validate split) — if not done
python scripts/eval_mimic.py --mode prepare --mimic_dir $MIMIC --reports_dir $MIMIC/reports \
  --output results/eval_val/ --split validate --unique_patients --max_samples 10

# Prepare test set (5 unique patients, test split) — if not done
python scripts/eval_mimic.py --mode prepare --mimic_dir $MIMIC --reports_dir $MIMIC/reports \
  --output results/eval_test/ --split test --unique_patients --max_samples 5

# Baselines on validation set (run once)
python scripts/eval_mimic.py --mode sonnet --output results/eval_val/ --config $CFG
python scripts/eval_mimic.py --mode chexone --output results/eval_val/ --config $CFG
python scripts/eval_mimic.py --mode score --output results/eval_val/

# Baselines on test set (run once — needed for auto-graduation comparison)
python scripts/eval_mimic.py --mode sonnet --output results/eval_test/ --config $CFG
python scripts/eval_mimic.py --mode chexone --output results/eval_test/ --config $CFG
python scripts/eval_mimic.py --mode score --output results/eval_test/
```

### Step 6: Initialize results.tsv and confirm

Create `results.tsv` with header row and baseline results. Do NOT commit this file.

```
commit	1/radcliq_v1	radgraph_f1	semb_score	bertscore_f1	bleu_2	ratescore	green_score	status	description
—	—	—	—	—	—	—	—	baseline	sonnet api vision-only
—	—	—	—	—	—	—	—	baseline	chexone direct
```

Once all 6 steps pass, begin the experiment loop.

## What you CAN modify

- `agent/prompts.py` — system prompt, templates, skill injection. **Primary lever.**
- `agent/react_agent.py` — ReAct loop, iteration count, tool selection strategy.
- `configs/config_grounded.yaml` — tool enablement, temperature, max_iterations.
- `skills/*.md` — clinical reasoning skills injected into the system prompt.

## What you CANNOT modify

- `scripts/eval_mimic.py` — evaluation harness. Scoring logic is frozen; CLI flags are OK to add.
- `tools/*.py` and `servers/*.py` — tool and server implementations.
- `clear/` — CLEAR concept scorer.
- `scripts/prepare_mimic_studies.py` — enriched data preparation.

## Data splits

**Validation set** (`results/eval_val/test_set.json`): 10 unique patients from MIMIC-CXR `validate` split. Use this for ALL iterative optimization. Prepared with `--split validate --unique_patients --max_samples 10`.

**Test set** (`results/eval_test/test_set.json`): 5 unique patients from MIMIC-CXR `test` split. Used for auto-graduation checkpoints (see below). Prepared with `--split test --unique_patients --max_samples 5`.

**Important**: Both sets use `--unique_patients` (one study per patient via `drop_duplicates('subject_id')`) to avoid data leakage.

## Running experiments

```bash
MIMIC=/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0
CFG=configs/config_grounded.yaml
ENRICHED=results/eval_enriched/val_studies_enriched.json

# Agent run on VALIDATION set (with prior study + clinical context)
python scripts/eval_mimic.py --mode agent --output results/eval_val_iter_N/ --config $CFG \
  --use_prior --use_clinical_context --enriched_json $ENRICHED

# Score with all 7 ReXrank metrics
python scripts/eval_mimic.py --mode score --output results/eval_val_iter_N/

# Compare against baselines
python scripts/eval_mimic.py --mode compare --output results/eval_val_iter_N/
```

Extract key metrics: `cat results/eval_val_iter_N/comparison.txt`

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated). Do NOT commit this file.

```
commit	1/radcliq_v1	radgraph_f1	semb_score	bertscore_f1	bleu_2	ratescore	green_score	status	description
a1b2c3d	—	—	—	—	—	—	—	keep	initial agent run with prior+context
```

Columns:
1. Git commit hash (short, 7 chars)
2-8. All 7 ReXrank metric values (use 0.0 for crashes)
9. Status: `keep`, `discard`, or `crash`
10. Short description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Read current state: `cat results.tsv`, check which metrics still lag baselines.
2. Hypothesize a targeted change. **One idea per iteration.** Examples:
   - "Reports too verbose → tighten word count in system prompt"
   - "Low RadGraph-F1 → agent missing entities → add more classification calls"
   - "Low BLEU-2 → wording diverges from radiology conventions → add style examples"
   - "Low RaTEScore → temporal/factual inconsistencies → strengthen FactCheXcker usage"
   - "Low GREEN → clinical errors (missed/false findings) → add more cross-checking between tools"
   - "Agent not using enough tools → add explicit instructions to call all 10"
   - "FactCheXcker always says OK → revise how the agent feeds the draft report"
   - "Prior study context not reflected → ensure skill references prior when available"
3. Implement: edit `agent/prompts.py`, `skills/*.md`, `configs/config_grounded.yaml`, or `agent/react_agent.py`.
4. git commit.
5. Run on VALIDATION:
   ```bash
   python scripts/eval_mimic.py --mode agent --output results/eval_val_iter_N/ --config $CFG \
     --use_prior --use_clinical_context --enriched_json $ENRICHED
   ```
6. Score: `python scripts/eval_mimic.py --mode score --output results/eval_val_iter_N/`
7. Read results: `cat results/eval_val_iter_N/comparison.txt`
8. Record in `results.tsv`.
9. If metrics improved → **keep** the commit, advance.
10. If metrics regressed → **discard**, `git reset --hard HEAD~1`.
11. Repeat.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**EvoTest** (Approach B): If manual iteration stalls after 5+ rounds with no improvement, switch to automated skill evolution:
```bash
python scripts/evotest_cxr.py --mode full --episodes 10 \
  --val-json results/eval_enriched/val_studies_enriched.json \
  --test-json results/eval_enriched/test_studies_enriched.json \
  --config $CFG --n-train 30 --n-test 100
```
UCB tree search + Evolver LLM generates improved skills in `skills/evo/`. After evotest, take the best skill and continue manual iteration from there.

**Crashes**: If a run crashes, check the error. If it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, log `crash` in results.tsv, revert, and move on.

**Timeout**: Each agent run on 10 validation CXRs should take ~10-30 minutes. If it exceeds 60 minutes, kill and discard.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human anything. Do NOT ask "should I continue?" or "is this good enough?". The human may be asleep. Continue indefinitely until manually interrupted. If you run out of ideas, think harder — re-read tool outputs, try combining approaches, try radical prompt changes, try different skill strategies. The loop runs until the human stops you.

**Auto-graduation**: After each iteration, check if ALL 7 val metrics beat ALL baselines. When they do:
1. Automatically run the 5-patient test set (see FINAL TEST below).
2. Log test results to `results.tsv` with status `test`.
3. **Keep iterating on val** — do NOT stop. The test run is a checkpoint, not a finish line. If a later iteration improves val further, run test again.

This way the human gets test results as soon as val looks good, but the agent never stops improving.

## FINAL TEST

Triggered automatically by auto-graduation, or manually by the human. Uses the same `results/eval_test/` directory that contains `test_set.json` and baseline predictions:

```bash
ENRICHED_TEST=results/eval_enriched/test_studies_enriched.json

# Run agent on 5-patient test set
python scripts/eval_mimic.py --mode agent --output results/eval_test/ --config $CFG \
  --use_prior --use_clinical_context --enriched_json $ENRICHED_TEST

# Score and compare against test baselines
python scripts/eval_mimic.py --mode score --output results/eval_test/
python scripts/eval_mimic.py --mode compare --output results/eval_test/
```

For subsequent auto-graduation runs, copy test_set.json + baseline files into `results/eval_test_iter_N/` to preserve history.

## Reference

- **MIMIC-CXR-JPG**: `/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/`
- **Config**: `configs/config_grounded.yaml`
- **Skill file**: `skills/grounded_report.md`
- **Enriched data**: `results/eval_enriched/val_studies_enriched.json`, `test_studies_enriched.json`
- **GPU 0**: CheXagent-2 (18.6GB) | **GPU 1**: CheXOne + CheXzero(:8009) + BiomedParse | **GPU 2**: FactCheXcker + eval | **CPU**: CXR Foundation(:8008)
- **Conda envs**: `cxr_agent` (main), `cxr_chexagent2`, `radgraph` (eval step 1: CXR-Report-Metric), `green_score` (eval steps 2-3: RaTEScore + GREEN)
- **ReXrank-metric**: `../ReXrank-metric/` — orchestration scripts for all 7 metrics
- **Server safety**: shared GPU server. Never delete outside `CXR_Agent/`, never touch other envs, never kill others' processes.
