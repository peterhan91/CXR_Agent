# CXR Agent — Day 1 Program

Autonomous setup and evaluation for the CXR Report Generation Agent on a multi-GPU server. Modeled after [autoresearch](https://github.com/karpathy/autoresearch): an agent reads this file, executes the steps, and reports results — no human in the loop.

**The goal is simple: produce grounded radiology reports that beat both Sonnet API and CheXOne on ALL ReXrank scores.**

## What Are Grounded Reports?

A grounded report links each textual finding to its spatial location in the CXR image via bounding boxes and/or segmentation masks. Instead of just text, the agent outputs:

```json
{
  "findings": "Cardiomegaly. Bilateral pleural effusions.",
  "impression": "Cardiomegaly with bilateral pleural effusions.",
  "grounding": [
    {
      "finding": "Cardiomegaly",
      "boxes": [[0.25, 0.30, 0.75, 0.85]],
      "mask": null
    },
    {
      "finding": "Left pleural effusion",
      "boxes": [[0.55, 0.70, 0.95, 0.98]],
      "mask": "base64-encoded RLE or PNG mask"
    }
  ]
}
```

The agent uses multiple grounding tools:
- **Bounding boxes**: `chexagent2_grounding` (phrase grounding), `medversa_detect` (abnormality detection). Normalized 0–1 `[x_min, y_min, x_max, y_max]`.
- **Segmentation masks**: `biomedparse_segment` (anatomical/pathology), `medsam3_segment` (text-guided SAM), `medversa_segment` (2D segmentation with coverage %). Pixel-level delineation for diffuse findings (opacities, effusions, pneumonia) where a bounding box is insufficient.

Each finding gets the most appropriate spatial grounding — focal findings (nodules, devices) use boxes; diffuse findings (effusions, opacities) use masks; some get both.

## Server Safety Rules (READ FIRST — NON-NEGOTIABLE)

You are operating on a **shared GPU server**. Other users have jobs, data, and environments on this machine. These rules are absolute — violating any of them could destroy someone else's work.

**NEVER:**
- Delete, move, or overwrite files **outside** `CXR_Agent/` and its designated sibling repos (`../MedVersa/`, `../BiomedParse/`, `../MedSAM3/`, `../FactCheXcker/`, `../cxr_concept/`)
- Run `rm -rf` on **any** directory. If you must delete, use targeted `rm` on specific files you created
- Modify or delete **any conda environment** except `cxr_agent`. Never touch base, other users' envs, or system Python
- Run `conda install` or `pip install` into base env or system Python. Always `conda activate cxr_agent` first
- Kill processes you didn't start. Before `pkill`/`kill`, verify the PID belongs to your session. Use `bash scripts/launch_servers.sh --stop` for our servers only
- Modify system files: `/etc/`, `/usr/`, `/opt/`, `~/.bashrc`, `~/.bash_profile`, `~/.zshrc`, or any system config
- Access or modify other users' home directories or data
- Change file permissions on shared directories (`chmod -R`, `chown`)
- Expose server ports to the public internet without confirmation. Binding to `0.0.0.0` is OK for local GPU cluster LAN
- Download large files (>1GB) without checking disk space first: `df -h`
- Run `git clean -fdx` or `git reset --hard` in sibling repos you cloned — these may have local modifications

**ALWAYS:**
- Activate `cxr_agent` env before any Python or pip command: `conda activate cxr_agent`
- Check disk space before large downloads: `df -h /path/to/data`
- Use the project's log directory (`logs/`) for server output
- Stop our servers before restarting: `bash scripts/launch_servers.sh --stop`
- Clean up temporary files you create in `tmp/` or `results/`
- Use absolute paths in server launch commands to avoid confusion
- Before cloning external repos, check if they already exist: `ls ../MedVersa`
- Verify CUDA availability before launching GPU servers: `python -c "import torch; print(torch.cuda.device_count())"`

## Setup

To set up the evaluation, work through these steps in order. Each step is **idempotent** — safe to re-run if something fails partway through.

### Step 1: Verify Environment ✅ DONE

<details><summary>Reference (click to expand)</summary>

```bash
conda activate cxr_agent
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
df -h .
cd /path/to/CXR_Agent
ls README.md
```
</details>

### Step 2: Install Dependencies ✅ DONE

<details><summary>Reference (click to expand)</summary>

```bash
conda activate cxr_agent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install anthropic tenacity pyyaml numpy h5py Pillow ftfy regex pandas
pip install transformers accelerate sentencepiece protobuf
pip install fastapi uvicorn requests
pip install qwen-vl-utils  # CheXOne (Qwen2.5-VL)
pip install nltk bert-score  # Evaluation
python -c "import anthropic, torch, fastapi; print('OK')"
```
</details>

### Step 3: Clone External Repos ✅ DONE

<details><summary>Reference (click to expand)</summary>

```bash
cd "$(dirname /path/to/CXR_Agent)"
[ ! -d MedVersa ] && git clone https://huggingface.co/hyzhou/MedVersa
[ ! -d BiomedParse ] && git clone https://github.com/microsoft/BiomedParse.git
[ ! -d MedSAM3 ] && git clone https://github.com/Joey-S-Liu/MedSAM3.git
[ ! -d FactCheXcker ] && git clone https://github.com/rajpurkarlab/FactCheXcker.git

conda activate cxr_agent
pip install git+https://github.com/facebookresearch/detectron2.git 2>/dev/null || true
[ -f ../FactCheXcker/requirements.txt ] && pip install -r ../FactCheXcker/requirements.txt
```
</details>

### Step 4: Validate Models

Validate each model loads and produces output. This auto-downloads HuggingFace weights on first run.

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

# Run all validations (downloads sample CXR if needed)
python scripts/validate_models/validate_all.py

# If some models fail, skip them and validate the rest:
python scripts/validate_models/validate_all.py --skip biomedparse medsam3 factchexcker

# Minimum viable: CheXagent-2 + CheXOne + CLEAR
python scripts/validate_models/validate_all.py --only chexagent2 chexone clear
```

### Step 5: Launch Servers

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

# Stop any existing servers first (safe, won't affect other processes)
bash scripts/launch_servers.sh --stop

# Launch all servers (6 processes across 3 GPUs)
bash scripts/launch_servers.sh

# Or core only (CheXagent-2 + CheXOne) if external repos aren't ready:
bash scripts/launch_servers.sh --only core
```

Wait for all servers to be healthy before proceeding:
```bash
# Poll health endpoints (retry a few times, models take 30-120s to load)
for port in 8001 8002 8004 8005 8006 8007; do
    for i in $(seq 1 30); do
        curl -sf http://localhost:$port/health > /dev/null && echo "Port $port: OK" && break
        sleep 5
    done
done
```

### Step 6: Test Individual Tools

Before running the full agent, test each tool individually with a real CXR image. This catches broken tools early and produces reference outputs. Save all results to `results/tool_validation/`.

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent
mkdir -p results/tool_validation

# Use any CXR image (validation scripts download a sample automatically)
IMG=/path/to/test_cxr.png

# --- CheXagent-2 (:8001) — 5 tools ---
curl -s -X POST http://localhost:8001/report \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/chexagent2_report.json

curl -s -X POST http://localhost:8001/srrg_report \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/chexagent2_srrg.json

curl -s -X POST http://localhost:8001/ground \
  -F "image=@$IMG" -F 'task=phrase_grounding' -F 'phrase=cardiomegaly' \
  | python -m json.tool | tee results/tool_validation/chexagent2_grounding.json

curl -s -X POST http://localhost:8001/classify \
  -F "image=@$IMG" -F 'task=binary_disease' -F 'disease=cardiomegaly' \
  | python -m json.tool | tee results/tool_validation/chexagent2_classify.json

curl -s -X POST http://localhost:8001/vqa \
  -F "image=@$IMG" -F 'question=Is there a pleural effusion?' \
  | python -m json.tool | tee results/tool_validation/chexagent2_vqa.json

# --- CheXOne (:8002) — 1 tool ---
curl -s -X POST http://localhost:8002/report \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/chexone_report.json

# --- MedVersa (:8004) — 5 tools ---
curl -s -X POST http://localhost:8004/report \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/medversa_report.json

curl -s -X POST http://localhost:8004/classify \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/medversa_classify.json

curl -s -X POST http://localhost:8004/detect \
  -F "image=@$IMG" | python -m json.tool | tee results/tool_validation/medversa_detect.json

curl -s -X POST http://localhost:8004/segment \
  -F "image=@$IMG" -F 'target=lungs' \
  | python -m json.tool | tee results/tool_validation/medversa_segment.json

curl -s -X POST http://localhost:8004/vqa \
  -F "image=@$IMG" -F 'question=Is there a pleural effusion?' \
  | python -m json.tool | tee results/tool_validation/medversa_vqa.json

# --- BiomedParse (:8005) — 1 tool ---
curl -s -X POST http://localhost:8005/segment \
  -F "image=@$IMG" -F 'target=lung opacity' \
  | python -m json.tool | tee results/tool_validation/biomedparse_segment.json

# --- MedSAM3 (:8006) — 1 tool ---
curl -s -X POST http://localhost:8006/segment \
  -F "image=@$IMG" -F 'target=pleural effusion' \
  | python -m json.tool | tee results/tool_validation/medsam3_segment.json

# --- FactCheXcker (:8007) — 1 tool ---
curl -s -X POST http://localhost:8007/verify \
  -F "image=@$IMG" -F 'report=The heart is enlarged. There is a 3.5cm nodule in the right lung.' \
  | python -m json.tool | tee results/tool_validation/factchexcker_verify.json
```

Check that every tool returned a valid JSON response (not an error). Review the outputs:
```bash
# Quick summary: list all files and check for errors
for f in results/tool_validation/*.json; do
    echo "=== $(basename $f) ==="
    python -c "import json; d=json.load(open('$f')); print('OK' if 'error' not in d else f'ERROR: {d[\"error\"]}')" 2>/dev/null || echo "INVALID JSON"
done
```

**Inspect outputs carefully — especially bounding boxes and segmentation masks:**

- **Bounding boxes** (from `chexagent2_grounding`, `medversa_detect`): Verify coordinates are normalized 0–1 and land on sensible anatomical regions. Boxes of `[0,0,0,0]` or `[0,0,1,1]` (full image) indicate parsing failures.
- **Segmentation masks** (from `biomedparse_segment`, `medsam3_segment`, `medversa_segment`): Verify masks are non-empty and cover plausible regions. An all-zeros mask or an all-ones mask means the model failed to segment.
- **Reports** (from `chexagent2_report`, `chexone_report`, `medversa_report`): Verify they contain actual radiology text, not empty strings or error messages.
- **Classifications** (from `chexagent2_classify`, `medversa_classify`): Verify they return expected label format (positive/negative, probabilities, etc.).

**If any output looks wrong or suboptimal:**
1. Check the server log in `logs/` for errors or warnings
2. Check the tool wrapper code in `tools/*.py` and server code in `servers/*.py` — look for input format mismatches (image encoding, parameter names, task strings)
3. Consult the model's original repo or HuggingFace page for correct API usage:
   - CheXagent-2: check `../cxr_concept/` and HuggingFace model card
   - CheXOne: check HuggingFace Qwen2.5-VL docs
   - MedVersa: check `../MedVersa/` repo — especially exact task parameter values
   - BiomedParse: check `../BiomedParse/` repo — supported target prompts
   - MedSAM3: check `../MedSAM3/` repo — text prompt format
   - FactCheXcker: check `../FactCheXcker/` repo
4. Fix the server or tool wrapper, restart the server, and re-test until the output is correct
5. Save the corrected outputs to `results/tool_validation/` — these serve as reference for what good output looks like

Do not proceed to the smoke test until all tools produce sensible outputs. A broken grounding or segmentation tool will silently degrade the agent's reports.

### Step 7: Smoke Test

Run the full agent on a single image to verify the end-to-end pipeline works, including grounding.

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

python scripts/run_agent.py --image /path/to/test_cxr.png --output results/smoke_test/

# Check output — verify it has findings, impression, AND grounding
cat results/smoke_test/*_result.json | python -m json.tool | head -50
```

If this produces a report with FINDINGS, IMPRESSION, and grounding bounding boxes, setup is complete.

## Evaluation — Establish Baselines

Once setup is verified, run baselines on MIMIC-CXR. We use **5 CXRs from the test set** for evaluation and **20-30 CXRs from the validation set** for skill evolution training.

### Step 1: Prepare MIMIC-CXR Splits

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

# Test set: 5 CXRs for evaluation (the numbers to beat)
python scripts/eval_mimic.py --mode prepare \
    --mimic_dir /path/to/mimic-cxr-jpg \
    --reports_dir /path/to/mimic-cxr/files \
    --output results/eval/ \
    --split test \
    --max_samples 5

# Validation set: 20-30 CXRs for skill evolution training
python scripts/eval_mimic.py --mode prepare \
    --mimic_dir /path/to/mimic-cxr-jpg \
    --reports_dir /path/to/mimic-cxr/files \
    --output results/eval_train/ \
    --split validate \
    --max_samples 30
```

**MIMIC-CXR paths**: Ask the human where the data lives on this server, or check common locations:
```bash
ls /data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv* 2>/dev/null
ls /data/physionet/mimic-cxr-jpg/ 2>/dev/null
ls ~/data/mimic-cxr-jpg/ 2>/dev/null
```

### Step 2: Run Baselines (on 5 test CXRs)

```bash
# Sonnet API baseline — vision-only, no tools, no CLEAR
python scripts/eval_mimic.py --mode sonnet --output results/eval/

# CheXOne baseline — direct server call, no agent, no CLEAR
python scripts/eval_mimic.py --mode chexone --output results/eval/
```

### Step 3: Run CXR Agent (initial, on 5 test CXRs)

```bash
python scripts/eval_mimic.py --mode agent --output results/eval/
```

### Step 4: Score and Compare

```bash
# Score all prediction files
python scripts/eval_mimic.py --mode score --output results/eval/

# For full ReXrank metrics:
pip install CXR-Report-Metric
python scripts/eval_mimic.py --mode score --output results/eval/

# Compare side-by-side
python scripts/eval_mimic.py --mode compare --output results/eval/
```

## The Experiment Loop

After establishing baselines, enter the improvement loop. The goal: **beat Sonnet API AND CheXOne on every ReXrank metric** (RadCliQ-v1, RadGraph-F1, SembScore, BERTScore, BLEU-2).

**Two approaches** — use either or both:

### Approach A: Manual Prompt Iteration

Direct edits to the agent's prompts and config, evaluated on the 5 test CXRs.

**What you CAN modify:**
- `agent/prompts.py` — system prompt, concept prior template, skill injection. This is the primary lever.
- `agent/react_agent.py` — ReAct loop behavior, iteration count, tool selection strategy, grounding extraction.
- `configs/config.yaml` — tool enablement, reasoning effort, max iterations, temperature.
- `skills/*.md` — add skill files that inject clinical reasoning strategies.

**What you CANNOT modify:**
- `scripts/eval_mimic.py` — the evaluation harness is fixed. It is the ground truth scorer.
- `tools/*.py` and `servers/*.py` — tool implementations and model servers are fixed.
- `clear/` — the CLEAR concept scorer is fixed.

LOOP:

1. **Check current state**: Read `results.tsv` and `results/eval/comparison.txt`. Identify which metrics still fall short of both baselines.
2. **Hypothesize**: Based on the gap, form a specific hypothesis about what to change. Examples:
   - "Agent reports are too verbose → tighten the system prompt to match MIMIC-CXR style"
   - "Low RadGraph-F1 → agent is missing clinical entities → increase tool calls for verification"
   - "Low BLEU-2 → agent wording diverges from radiology conventions → add style constraints to prompt"
   - "Agent isn't using grounding tools enough → add explicit instruction to ground every finding"
3. **Implement**: Edit the relevant file(s). Keep changes minimal and targeted — one hypothesis per iteration.
4. **Eval on 5 test CXRs**:
   ```bash
   python scripts/eval_mimic.py --mode agent --output results/eval_iter_N/
   python scripts/eval_mimic.py --mode score --output results/eval_iter_N/
   ```
5. **Record**: Append results to `results.tsv`.
6. **Decide**:
   - If ALL metrics improved or held steady with at least one improving → **keep** the change, git commit.
   - If any metric regressed significantly → **discard**, revert to last good state.
   - If mixed results → use judgment. A small regression in one metric may be acceptable if others improved substantially.
7. **Repeat** until all 5 metrics beat both baselines.

### Approach B: EvoTest Skill Evolution (via `../mimic_skills`)

Automated skill evolution using the UCB tree search from `../mimic_skills/EvoTest/`. This trains on 20-30 validation CXRs and tests on the 5 test CXRs. Read `../mimic_skills/` to understand the full framework before using it.

**How it works:**
1. The Evolver LLM analyzes agent failure trajectories on the validation set (missed findings, hallucinations, poor style, weak grounding)
2. It generates an improved skill (clinical reasoning strategy as markdown, injected into the system prompt via `SKILL_INJECTION_TEMPLATE` in `agent/prompts.py`)
3. The agent runs with the new skill on the validation set, scores are computed
4. UCB tree search selects the best-performing skill branch and evolves further
5. Best skill is saved to `skills/` and evaluated on the 5 test CXRs

**Key files in `../mimic_skills/`:**
- `EvoTest/src/our_agent.py` — UCB tree + prompt evolution logic
- `EvoTest/src/cross_episode_memory.py` — positive/negative example persistence across episodes
- `scripts/evolve_skill.py` — Evolver LLM: generates skill from trajectories + ground truth
- `scripts/evotest_clinical.py` — full episode loop (adapt this for CXR report generation)
- `skills/` — example evolved skill files (markdown format)

**Adaptation for CXR reports (5 episodes max):**
```
TRAIN: 20-30 CXRs from MIMIC-CXR validation set (results/eval_train/)
TEST:  5 CXRs from MIMIC-CXR test set (results/eval/)
EPISODES: 5 max (each episode ≈ 15-30 min, total ≈ 1.5-2.5 hours)

Episode 0: BASELINE — run agent without any skill on validation set
Episodes 1-4: EVOLUTION
  1. Select parent skill via UCB tree
  2. Evolver generates improved skill based on:
     - Agent trajectories on validation CXRs (tool calls, reasoning, final report)
     - Ground truth reports (FINDINGS + IMPRESSION)
     - Failure analysis (which findings were missed/hallucinated, style gaps)
  3. Run agent with new skill on validation set → compute ReXrank scores
  4. Update UCB tree node with scores
Episode 5: FINAL — take best skill → eval on 5 test CXRs
  Save winning skill to skills/cxr_evolved_vN.md
```

**Skill injection is already wired up:**
- `agent/prompts.py` has `build_skills_prompt()` and `SKILL_INJECTION_TEMPLATE`
- `agent/react_agent.py` accepts `skill_text` parameter
- `skills/` directory is ready for evolved skill files

**The metrics (all 5 must improve over BOTH baselines on the 5 test CXRs):**
1. **RadCliQ-v1** (primary) — composite clinical quality score (lower is better)
2. **RadGraph-F1** — clinical entity and relation overlap (higher is better)
3. **SembScore** — sentence embedding similarity (higher is better)
4. **BERTScore** — contextual embedding F1 (higher is better)
5. **BLEU-2** — bigram precision (higher is better)

## Results Logging

Log every experiment to `results.tsv` (tab-separated). Do NOT commit this file — keep it untracked.

```
iter	commit	mode	num_studies	split	radcliq_v1	radgraph_f1	semb_score	bertscore_f1	bleu_2	status	description
0	—	sonnet	5	test	—	—	—	0.4200	0.0900	baseline	Sonnet API vision-only
0	—	chexone	5	test	—	—	—	0.4521	0.1234	baseline	CheXOne direct, no agent
1	a1b2c3d	agent	5	test	—	—	—	0.4600	0.1100	keep	initial agent run
2	b2c3d4e	agent	30	val	—	—	—	0.4750	0.1300	keep	evotest episode 1 (train)
2	b2c3d4e	agent	5	test	—	—	—	0.4720	0.1280	keep	evotest episode 1 (eval)
3	c3d4e5f	agent	5	test	—	—	—	0.4680	0.1250	discard	manual prompt tweak (hurt BLEU)
```

## Checkpoints

Check what's already been done:

```bash
# Test set (5 CXRs) prepared?
ls results/eval/test_set.json

# Validation set (20-30 CXRs) prepared?
ls results/eval_train/test_set.json

# Baseline predictions (on 5 test CXRs)?
python -c "import json; d=json.load(open('results/eval/predictions_sonnet.json')); print(f'Sonnet: {len(d)}')" 2>/dev/null
python -c "import json; d=json.load(open('results/eval/predictions_chexone.json')); print(f'CheXOne: {len(d)}')" 2>/dev/null

# Agent predictions?
python -c "import json; d=json.load(open('results/eval/predictions_agent.json')); print(f'Agent: {len(d)}')" 2>/dev/null

# Evolved skills?
ls skills/*.md 2>/dev/null

# Scores and comparison?
ls results/eval/scores_*.json 2>/dev/null
cat results/eval/comparison.txt 2>/dev/null
cat results.tsv 2>/dev/null
```

Skip any step where the output already exists and looks complete.

## Error Recovery

- **Server crashed**: Check logs in `logs/*.log`, restart with `bash scripts/launch_servers.sh`
- **Agent failed on a study**: The eval script skips failures and continues. Check `errors` count in output
- **Prediction run interrupted**: Just re-run the same command — it resumes from the last saved checkpoint
- **Disk full**: Check `df -h`, clean up `logs/` and `tmp/`. Do NOT delete `results/`
- **OOM on GPU**: Check `nvidia-smi`. If a model server OOMed, restart it. Consider launching fewer servers
- **API rate limit**: The agent uses exponential backoff (tenacity). It will retry automatically
- **Grounding tool returns no boxes**: Some findings may not be localizable. The agent should gracefully handle this — report the finding without grounding rather than omitting it

## Notes

- **Test/train split discipline**: Always train/evolve on the 20-30 validation CXRs (`results/eval_train/`), always evaluate on the 5 test CXRs (`results/eval/`). Never tune on the test set.
- All eval runs are **resume-safe**: predictions are saved incrementally, and completed studies are skipped on re-run
- Sonnet API baseline requires only ANTHROPIC_API_KEY. CheXOne baseline requires only the CheXOne server (:8002). Agent mode requires all configured servers
- Token usage is tracked per-study in agent predictions for cost monitoring
- **EvoTest reference**: `../mimic_skills/` contains the full evotest framework — read it before attempting Approach B. The UCB tree search, failure analysis, and skill generation patterns there are directly applicable to CXR report generation.
- **Grounded reports are a bonus output** — ReXrank scores are computed on text only (FINDINGS + IMPRESSION). Grounding adds clinical value but does not directly affect the metric scores. Focus on text quality first, then ensure grounding is correct
