# CXR Agent — Day 1 Program

Autonomous setup and evaluation for the CXR Report Generation Agent on a multi-GPU server. Modeled after [autoresearch](https://github.com/karpathy/autoresearch): an agent reads this file, executes the steps, and reports results — no human in the loop.

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

### Step 1: Verify Environment

Check the basics before doing anything else.

```bash
# Conda env exists?
conda env list | grep cxr_agent

# If not, create it:
conda create -n cxr_agent python=3.10 -y

# Activate
conda activate cxr_agent

# GPUs available?
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Disk space OK? (need ~50GB free for models)
df -h .

# Project root
cd /path/to/CXR_Agent
ls README.md  # sanity check
```

### Step 2: Install Dependencies

```bash
conda activate cxr_agent

# PyTorch + CUDA (skip if already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install anthropic tenacity pyyaml numpy h5py Pillow ftfy regex pandas
pip install transformers accelerate sentencepiece protobuf
pip install fastapi uvicorn requests

# CheXOne (Qwen2.5-VL)
pip install qwen-vl-utils

# Evaluation
pip install nltk bert-score

# Verify key imports
python -c "import anthropic, torch, fastapi; print('OK')"
```

### Step 3: Clone External Repos (if needed)

Only clone repos that don't already exist. **Never force-clone over existing repos.**

```bash
cd "$(dirname /path/to/CXR_Agent)"  # parent directory

# MedVersa
[ ! -d MedVersa ] && git clone https://huggingface.co/hyzhou/MedVersa

# BiomedParse
[ ! -d BiomedParse ] && git clone https://github.com/microsoft/BiomedParse.git

# MedSAM3
[ ! -d MedSAM3 ] && git clone https://github.com/Joey-S-Liu/MedSAM3.git

# FactCheXcker
[ ! -d FactCheXcker ] && git clone https://github.com/rajpurkarlab/FactCheXcker.git
```

Install repo-specific dependencies (into cxr_agent env):
```bash
conda activate cxr_agent

# BiomedParse: detectron2
pip install git+https://github.com/facebookresearch/detectron2.git 2>/dev/null || echo "detectron2 already installed or failed (optional)"

# FactCheXcker
[ -f ../FactCheXcker/requirements.txt ] && pip install -r ../FactCheXcker/requirements.txt
```

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

### Step 6: Smoke Test

Run the agent on a single image to verify the full pipeline works.

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

# Use any CXR image — validation scripts download a sample automatically
python scripts/run_agent.py --image /path/to/test_cxr.png --output results/smoke_test/

# Check output
cat results/smoke_test/*_result.json | python -m json.tool | head -30
```

If this produces a report with FINDINGS and IMPRESSION sections, setup is complete.

## Evaluation

Once setup is verified, run the MIMIC-CXR evaluation pipeline. This compares CheXOne (baseline) vs CXR Agent (ours).

### Step 1: Prepare MIMIC-CXR Test Set

```bash
cd /path/to/CXR_Agent
conda activate cxr_agent

python scripts/eval_mimic.py --mode prepare \
    --mimic_dir /path/to/mimic-cxr-jpg \
    --reports_dir /path/to/mimic-cxr/files \
    --output results/eval/

# For a quick test run (50 samples first):
python scripts/eval_mimic.py --mode prepare \
    --mimic_dir /path/to/mimic-cxr-jpg \
    --reports_dir /path/to/mimic-cxr/files \
    --output results/eval/ \
    --max_samples 50
```

This reads the official MIMIC-CXR split and metadata CSVs, selects frontal (PA/AP) test images, loads ground truth reports, and saves `results/eval/test_set.json`.

**MIMIC-CXR paths**: Ask the human where the data lives on this server, or check common locations:
```bash
ls /data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv* 2>/dev/null
ls /data/physionet/mimic-cxr-jpg/ 2>/dev/null
ls ~/data/mimic-cxr-jpg/ 2>/dev/null
```

### Step 2: Run CheXOne Baseline

```bash
python scripts/eval_mimic.py --mode chexone --output results/eval/
```

This calls the CheXOne server directly for each test image — no agent, no CLEAR prior. Predictions are saved incrementally (resume-safe). ~2-5 seconds per study.

### Step 3: Run CXR Agent

```bash
# Start with a small subset to verify it works:
python scripts/eval_mimic.py --mode prepare \
    --mimic_dir /path/to/mimic-cxr-jpg \
    --reports_dir /path/to/mimic-cxr/files \
    --output results/eval_pilot/ \
    --max_samples 20

python scripts/eval_mimic.py --mode agent --output results/eval_pilot/

# If pilot looks good, run the full test set:
python scripts/eval_mimic.py --mode agent --output results/eval/
```

This runs the full CXR Agent pipeline: CLEAR concept scoring + 14 tools + ReAct agent. ~30-60 seconds per study. Resume-safe (saves every 5 studies).

**Cost estimate**: ~$0.01-0.05 per study in Anthropic API costs. Full test set (~2,300 studies) = ~$25-120 total.

### Step 4: Score Predictions

```bash
python scripts/eval_mimic.py --mode score --output results/eval/
```

Scores all prediction files found in the output directory. Uses CXR-Report-Metric if installed, otherwise falls back to BLEU and BERTScore.

For full ReXrank metrics (RadCliQ-v1, RadGraph-F1, SembScore):
```bash
# Install CXR-Report-Metric (requires RadGraph PhysioNet access)
pip install CXR-Report-Metric
# Then re-run scoring
python scripts/eval_mimic.py --mode score --output results/eval/
```

### Step 5: Compare Results

```bash
python scripts/eval_mimic.py --mode compare --output results/eval/
```

Prints a side-by-side metric comparison table and saves it to `results/eval/comparison.txt`.

## Results Logging

After each major milestone, log results to `results.tsv` (tab-separated):

```
mode	num_studies	radcliq_v1	radgraph_f1	bertscore_f1	bleu_2	status	notes
chexone	2347	—	—	0.4521	0.1234	done	baseline, no CLEAR, no agent
agent	2347	—	—	0.4876	0.1567	done	full pipeline, CLEAR + 14 tools
agent_pilot	20	—	—	0.4712	0.1456	done	pilot run for sanity check
```

## Checkpoints

You can check what's already been done by looking at existing files:

```bash
# Test set prepared?
ls results/eval/test_set.json

# How many CheXOne predictions?
python -c "import json; d=json.load(open('results/eval/predictions_chexone.json')); print(len(d))" 2>/dev/null

# How many Agent predictions?
python -c "import json; d=json.load(open('results/eval/predictions_agent.json')); print(len(d))" 2>/dev/null

# Scores computed?
ls results/eval/scores_*.json 2>/dev/null

# Comparison available?
cat results/eval/comparison.txt 2>/dev/null
```

Skip any step where the output already exists and looks complete.

## Error Recovery

- **Server crashed**: Check logs in `logs/*.log`, restart with `bash scripts/launch_servers.sh`
- **Agent failed on a study**: The eval script skips failures and continues. Check `errors` count in output
- **Prediction run interrupted**: Just re-run the same command — it resumes from the last saved checkpoint
- **Disk full**: Check `df -h`, clean up `logs/` and `tmp/`. Do NOT delete `results/`
- **OOM on GPU**: Check `nvidia-smi`. If a model server OOMed, restart it. Consider launching fewer servers
- **API rate limit**: The agent uses exponential backoff (tenacity). It will retry automatically

## Notes

- All eval runs are **resume-safe**: predictions are saved incrementally, and completed studies are skipped on re-run
- CheXOne baseline requires only the CheXOne server (:8002). Agent mode requires all configured servers
- The agent pilot run (20 studies) is strongly recommended before the full run to catch configuration issues early
- Token usage is tracked per-study in agent predictions for cost monitoring
