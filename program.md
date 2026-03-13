# CXR Agent — Evaluation

Autonomous evaluation of the CXR report generation agent across 5 datasets. Run all phases sequentially — no human in the loop.

## Goal

**Evaluate the CXR agent on 100 studies** (50 baseline + 50 follow-up from `data/eval/sample_100.json`). Measure all 7 ReXrank metrics. Compare against baselines (CheXOne-R1, MedVersa-Internal). Run ablations (with/without CLEAR).

| Metric | Column | Range | Higher is Better | Measures |
|--------|--------|-------|-------------------|----------|
| BLEU-2 | bleu_score | [0, 1] | Yes | Bigram overlap (weights: 1/2, 1/2) via fast_bleu |
| BERTScore | bertscore | [-1, 1] | Yes | Contextual embedding F1 (distilroberta-base, baseline rescaling, no IDF) |
| SembScore | semb_score | [-1, 1] | Yes | Cosine similarity of CheXbert embeddings |
| RadGraph | radgraph_combined | [0, 1] | Yes | Mean of entity F1 and relation F1 from RadGraph (DyGIE++) |
| 1/RadCliQ-v1 | 1/RadCliQ-v1 | (0, +inf) | Yes | Inverse of RadCliQ-v1 composite (primary ranking metric) |
| RaTEScore | ratescore | [0, 1] | Yes | Factual/temporal consistency score |
| GREEN | green_score | [0, 1] | Yes | LLM-based clinical error analysis (StanfordAIMI/GREEN-radllama2-7b) |

Per study, output: report text (FINDINGS + IMPRESSION), grounded findings (JSON with bboxes), two figures (`{id}_bbox.png`, `{id}_mask.png`).

## Setup

Before starting evaluation, complete ALL steps below. Every step MUST pass. If any step fails, fix it before moving on — do NOT skip steps or proceed with a degraded setup. The human may be asleep.

### Step 1: Verify all 7 tool servers are alive

```bash
FAIL=0
for port in 8001 8002 8005 8007 8008 8009 8010; do
  STATUS=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:$port/health)
  if [ "$STATUS" = "200" ]; then
    echo "Port $port: OK"
  else
    echo "Port $port: FAIL (HTTP $STATUS)"
    FAIL=1
  fi
done
[ "$FAIL" = "0" ] || { echo "BLOCKED: restart failed servers before continuing"; exit 1; }
```

If a server is down, start it using the commands in "Server startup commands" at the bottom of this file. Wait for it to load (can take 30-120s for model loading), then re-check.

| Port | Server | Tools |
|------|--------|-------|
| 8001 | CheXagent-2 | `chexagent2_report`, `chexagent2_srrg_report`, `chexagent2_grounding`, `chexagent2_classify`, `chexagent2_vqa` |
| 8002 | CheXOne | `chexone_report` |
| 8005 | BiomedParse | `biomedparse_segment` |
| 8007 | FactCheXcker | `factchexcker_verify` |
| 8008 | CXR Foundation | `cxr_foundation_classify` |
| 8009 | CheXzero | `chexzero_classify` |
| 8010 | MedGemma | `medgemma_vqa`, `medgemma_report` |

Plus **CLEAR concept prior** — DINOv2+CLIP cosine similarity to MIMIC-CXR concepts, injected before tool calls.

**Disabled as agent tools**: MedVersa (hallucinating), MedSAM (poor CXR masks), MedSAM3 (replaced). Note: MedVersa-Internal is still used as a standalone baseline in Phase 3 (direct report generation, not as an agent tool).

### Step 2: Run a real inference through every server

Health endpoints only confirm the process is alive — not that the model is loaded or inference works. Send a real request to each server using a test image and verify the response is non-empty and well-formed.

Use any frontal CXR as test image. Example: first baseline study from `data/eval/sample_100.json`.

```bash
TEST_IMG=$(python3 -c "import json; d=json.load(open('data/eval/sample_100.json')); print(d['baseline'][0]['image_path'])")

# 1. CheXagent-2 report (port 8001)
curl -sf -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\", \"task\": \"report_generation\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert len(r.get('report',''))>20, f'Empty report: {r}'; print('8001 report: OK')"

# 2. CheXagent-2 grounding (port 8001)
curl -sf -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\", \"task\": \"phrase_grounding\", \"phrase\": \"heart\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); print('8001 grounding: OK')"

# 3. CheXOne report (port 8002)
curl -sf -X POST http://localhost:8002/generate_report \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert len(r.get('report',''))>20, f'Empty report: {r}'; print('8002 report: OK')"

# 4. BiomedParse segmentation (port 8005)
curl -sf -X POST http://localhost:8005/segment \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\", \"prompts\": [\"left lung\"]}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert 'results' in r or 'masks' in r, f'Bad response: {list(r.keys())}'; print('8005 segment: OK')"

# 5. FactCheXcker (port 8007) — test with ETT claim to verify full pipeline
curl -sf -X POST http://localhost:8007/verify_report \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\", \"report\": \"Endotracheal tube tip is 2 cm above the carina.\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'8007 verify: OK (changes_made={r.get(\"changes_made\",\"?\")})')"

# 6. CXR Foundation classify (port 8008)
curl -sf -X POST http://localhost:8008/classify \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert 'predictions' in r or 'results' in r, f'Bad response: {list(r.keys())}'; print('8008 classify: OK')"

# 7. CheXzero classify (port 8009)
curl -sf -X POST http://localhost:8009/classify \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert 'predictions' in r or 'results' in r, f'Bad response: {list(r.keys())}'; print('8009 classify: OK')"

# 8. MedGemma VQA (port 8010)
curl -sf -X POST http://localhost:8010/vqa \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$TEST_IMG\", \"question\": \"Is there a pleural effusion?\"}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert len(r.get('answer',''))>0, f'Empty answer: {r}'; print('8010 vqa: OK')"
```

**Every single check must print OK.** If any fails: check the server logs, restart the server, and re-run. Do NOT proceed with broken servers.

**Note on FactCheXcker**: The full pipeline verifies ETT/carina positioning only (config limited to ETT/carina objects). For general finding verification (effusions, cardiomegaly, etc.), use CheXzero + CXR Foundation classifiers and MedGemma VQA.

### Step 3: Verify ReXrank scoring pipeline

All 7 metrics must be computable. Test each scoring component:

```bash
# 1. CXR-Report-Metric (5 core metrics: BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ-v1)
cd /home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric && \
  conda run -n radgraph python -c "from CXRMetric.run_eval import calc_metric; print('CXR-Report-Metric: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent

# 2. RaTEScore
conda run -n green_score python -c "from RaTEScore import RaTEScore; print('RaTEScore: OK')"

# 3. GREEN (loads 7B model — takes ~30s first time)
cd /home/than/DeepLearning/GREEN && \
  conda run -n green_score python -c "from green_score.green import GREEN; print('GREEN: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent
```

If any fail, fix the conda environment (`radgraph` for step 1, `green_score` for steps 2-3) before proceeding. Scoring with missing metrics is NOT acceptable — all 7 are required.

### Step 4: Verify eval data files exist

```bash
FAIL=0
for f in data/eval/sample_100.json data/eval/mimic_cxr_test.json data/eval/chexpert_plus_valid.json \
         data/eval/rexgradient_test.json data/eval/iu_xray_test.json data/eval/padchest_gr_test.json; do
  [ -f "$f" ] && echo "$f: OK" || { echo "$f: MISSING"; FAIL=1; }
done
[ "$FAIL" = "0" ] || { echo "BLOCKED: regenerate missing data files"; exit 1; }
```

If eval data is missing, regenerate with `python scripts/prepare_eval_datasets.py --datasets all`.

Note: Prior study data (image + report) is embedded in `prior_study` field of each follow-up entry — no separate enriched JSON needed for the 100-study eval.

### Step 5: Verify Anthropic API key

The agent uses Claude Sonnet via the Anthropic API. Confirm the key works:

```bash
python3 -c "
import anthropic
c = anthropic.Anthropic()
r = c.messages.create(model='claude-sonnet-4-20250514', max_tokens=10, messages=[{'role':'user','content':'Say OK'}])
print(f'Anthropic API: OK (model={r.model})')
"
```

### Step 6: Dry-run the agent on 1 study

Run the full agent pipeline on a single study to confirm everything is wired end-to-end:

```bash
# Pick first baseline study from sample_100
python3 -c "
import json
d = json.load(open('data/eval/sample_100.json'))
json.dump({'baseline': [d['baseline'][0]], 'followup': []}, open('/tmp/test_1.json', 'w'))
print(f'Test study: {d[\"baseline\"][0][\"study_id\"]}')
"

# Run agent on that single study
python scripts/eval_mimic.py --mode agent --input /tmp/test_1.json --track baseline \
  --output /tmp/eval_dryrun/ --config configs/config_grounded.yaml

# Verify output exists
ls /tmp/eval_dryrun/predictions_agent.json && echo "Dry run: OK" || echo "Dry run: FAIL"
```

If this fails, debug the error before proceeding to batch runs. Common issues:
- Tool server timeout → restart the server
- API rate limit → wait and retry
- Missing config key → check `configs/config_grounded.yaml`

Once ALL 6 steps pass, proceed to the evaluation phases.

## Data splits

### Large-scale eval (5 datasets, 9,488 studies)

Prepared by `scripts/prepare_eval_datasets.py --datasets all`. Stored in `data/eval/`.

| Dataset | File | Total | Baseline | Follow-up |
|---------|------|------:|:--------:|:---------:|
| MIMIC-CXR | `mimic_cxr_test.json` | 2,210 | 43 | 2,167 |
| CheXpert-Plus | `chexpert_plus_valid.json` | 200 | 200 | 0 |
| ReXGradient | `rexgradient_test.json` | 5,573 | 3,504 | 2,069 |
| IU-Xray | `iu_xray_test.json` | 590 | 590 | 0 |
| PadChest-GR | `padchest_gr_test.json` | 915 | 915 | 0 |
| **TOTAL** | | **9,488** | **5,252** | **4,236** |

Each study has `eval_track` field ("baseline" or "followup"). Every follow-up has `prior_study` with image path + report. Every baseline has `prior_study=None`.

**Testing sample** (`data/eval/sample_100.json`): 50 baseline + 50 follow-up from unique patients (seed=42). Use this for cost-effective testing before full-scale runs.

## Evaluation — run all phases sequentially

After setup passes, run each phase in order. If a phase crashes, fix and retry that phase — do NOT skip it. The human may be asleep.

### Phase 1: Agent on baseline CXR (50 studies)

Single CXR, no prior context. Core evaluation.

```bash
CFG=configs/config_grounded.yaml

python scripts/eval_mimic.py --mode agent --input data/eval/sample_100.json --track baseline \
  --output results/eval_baseline/ --config $CFG
python scripts/eval_mimic.py --mode score --output results/eval_baseline/
```

Print scores when done: `cat results/eval_baseline/scores_agent.json`

### Phase 2: Agent on follow-up CXR (50 studies)

Current CXR + prior image + prior report. Tests comparison/temporal reasoning.

```bash
python scripts/eval_mimic.py --mode agent --input data/eval/sample_100.json --track followup \
  --output results/eval_followup/ --config $CFG
python scripts/eval_mimic.py --mode score --output results/eval_followup/
```

### Phase 3: Baselines on same 50 baseline studies

Run CheXOne-R1 and MedVersa-Internal on the same baseline studies for head-to-head comparison.

```bash
python scripts/eval_mimic.py --mode chexone --input data/eval/sample_100.json --track baseline \
  --output results/eval_baseline/
python scripts/eval_mimic.py --mode medversa --input data/eval/sample_100.json --track baseline \
  --output results/eval_baseline/
python scripts/eval_mimic.py --mode score --output results/eval_baseline/
python scripts/eval_mimic.py --mode compare --output results/eval_baseline/
```

### Phase 4: CLEAR ablation

Run agent with CLEAR concept scorer disabled on both tracks to measure CLEAR's contribution.

```bash
python scripts/eval_mimic.py --mode agent --input data/eval/sample_100.json --track baseline \
  --output results/eval_baseline_noclear/ --config $CFG --no_clear
python scripts/eval_mimic.py --mode score --output results/eval_baseline_noclear/

python scripts/eval_mimic.py --mode agent --input data/eval/sample_100.json --track followup \
  --output results/eval_followup_noclear/ --config $CFG --no_clear
python scripts/eval_mimic.py --mode score --output results/eval_followup_noclear/
```

### Phase 5: Save final summary

After all phases complete, save a consolidated results table to `results/eval_summary.json` and `results/eval_summary.txt`. Include:
- All 7 metrics for each method (agent, chexone, medversa) on baseline track
- All 7 metrics for agent on follow-up track
- All 7 metrics for agent with/without CLEAR on both tracks
- Per-study cost (input/output tokens, wall time, number of ReAct steps)

Also print the summary table to stdout.

### Error recovery

- If a server dies mid-run: restart it (see "Server startup commands"), re-run Step 2 health check, then retry the phase.
- If API rate-limited: wait 60s and retry.
- NEVER proceed to the next phase with partial results — the full batch must succeed.
- After all phases complete, print a summary of all scores.

## Reference

- **MIMIC-CXR-JPG**: `/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/`
- **MIMIC-IV**: `/home/than/physionet.org/files/mimiciv/3.1/`
- **Config**: `configs/config_grounded.yaml`
- **Skill file**: `skills/grounded_report.md`
- **Enriched data**: `results/eval_enriched/val_studies_enriched.json`, `test_studies_enriched.json`
- **GPU 0**: CheXagent-2 (18.6GB) | **GPU 1**: CheXOne + CheXzero(:8009) + BiomedParse | **GPU 2**: MedGemma(:8010, 8GB) + FactCheXcker + eval | **CPU**: CXR Foundation(:8008)
- **Conda envs**: `cxr_agent` (main, torch 2.6+cu124), `cxr_chexagent2`, `radgraph` (eval step 1: CXR-Report-Metric), `green_score` (eval steps 2-3: RaTEScore + GREEN)
- **ReXrank-metric**: `../ReXrank-metric/` — orchestration scripts for all 7 metrics
- **FactCheXcker**: `../FactCheXcker/` — LLM backend uses OpenAI API (OPENAI_API_KEY env var), verifies ETT/carina only
- **Server safety**: shared GPU server. Never delete outside `CXR_Agent/`, never touch other envs, never kill others' processes.

## Server startup commands

```bash
# GPU 0: CheXagent-2
conda run -n cxr_chexagent2 python servers/chexagent2_server.py --port 8001

# GPU 1: CheXOne + BiomedParse + CheXzero
CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002
CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005
CUDA_VISIBLE_DEVICES=1 python servers/chexzero_server.py --port 8009

# GPU 2: MedGemma + FactCheXcker
CUDA_VISIBLE_DEVICES=2 python servers/medgemma_server.py --port 8010
OPENAI_API_KEY="..." WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python servers/factchexcker_server.py --port 8007 --factchexcker_dir ../FactCheXcker

# CPU: CXR Foundation
python servers/cxr_foundation_server.py --port 8008
```
