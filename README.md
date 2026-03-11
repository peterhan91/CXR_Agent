# CXR Agent

ReAct agent for chest X-ray report generation. Claude Sonnet 4.6 orchestrates 14 specialized CXR tools via Anthropic native tool-use, with CLEAR concept priors as structured context.

## Architecture

```
                          Anthropic API
                              |
                     Claude Sonnet 4.6 (ReAct)
                      /       |        \
                CLEAR prior  Tools(14)  Adaptive Thinking
                (in-process)  (HTTP)    (effort: medium)
                     |
    ┌────────────────┼────────────────────┐
    GPU 0            GPU 1                GPU 2
    CheXagent-2      CheXOne              MedVersa
    :8001 (5 tools)  :8002 (1 tool)       :8004 (5 tools)
                     BiomedParse          MedSAM3
                     :8005 (1 tool)       :8006 (1 tool)
                                          FactCheXcker
                                          :8007 (1 tool)
```

**How it works**: For each CXR image, the agent (1) receives CLEAR concept similarity scores as a prior, (2) calls tools autonomously via ReAct to gather evidence, and (3) synthesizes a final FINDINGS + IMPRESSION report. No hardcoded workflow — Sonnet decides which tools to call and in what order based on tool descriptions and image content.

## Tools (14 total)

| Tool | Server | Capability |
|------|--------|------------|
| `chexagent2_report` | :8001 | Free-text CXR report generation |
| `chexagent2_srrg_report` | :8001 | Structured report by anatomy (Lungs, Pleura, CV, Other) |
| `chexagent2_grounding` | :8001 | Bounding boxes for findings, tubes, fractures, devices |
| `chexagent2_classify` | :8001 | View classification, binary disease, disease identification |
| `chexagent2_vqa` | :8001 | Visual question answering on CXR |
| `chexone_report` | :8002 | Report generation with optional reasoning trace |
| `medversa_report` | :8004 | Report generation with patient context (7B) |
| `medversa_classify` | :8004 | Classification across 33 pathologies |
| `medversa_detect` | :8004 | Abnormality detection with bounding boxes |
| `medversa_segment` | :8004 | 2D segmentation with coverage percentage |
| `medversa_vqa` | :8004 | Visual question answering (7B) |
| `biomedparse_segment` | :8005 | Anatomical/pathology segmentation (lungs, opacity, pneumonia) |
| `medsam3_segment` | :8006 | Text-guided SAM segmentation (broad vocabulary) |
| `factchexcker_verify` | :8007 | Verify/correct measurement hallucinations in reports |

## Quick Start (GPU Server)

### Prerequisites

- 3x NVIDIA A6000 (or equivalent, ~48GB VRAM each)
- ANTHROPIC_API_KEY set in environment
- CLEAR model checkpoint at `../cxr_concept/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt`
- CLEAR concepts at `../cxr_concept/concepts/mimic_concepts.csv` (368K MIMIC-CXR observations)

### 1. Install

```bash
conda create -n cxr_agent python=3.10 -y
conda activate cxr_agent
pip install -r requirements.txt

# For CheXOne (Qwen2.5-VL)
pip install qwen-vl-utils

# For BiomedParse
pip install git+https://github.com/facebookresearch/detectron2.git
```

See `scripts/validate_models/GPU_SERVER_SETUP.md` for full environment setup including external repos (MedVersa, BiomedParse, MedSAM3, FactCheXcker).

### 2. Launch Model Servers

```bash
# All servers (6 processes across 3 GPUs)
bash scripts/launch_servers.sh

# Core only (CheXagent-2 + CheXOne)
bash scripts/launch_servers.sh --only core

# Stop all
bash scripts/launch_servers.sh --stop
```

Wait for all servers to be healthy:

```bash
curl http://localhost:8001/health  # CheXagent-2
curl http://localhost:8002/health  # CheXOne
curl http://localhost:8004/health  # MedVersa
curl http://localhost:8005/health  # BiomedParse
curl http://localhost:8006/health  # MedSAM3
curl http://localhost:8007/health  # FactCheXcker
```

### 3. Validate Models

```bash
# All models
python scripts/validate_models/validate_all.py

# Specific models
python scripts/validate_models/validate_all.py --only chexagent2 chexone clear

# Skip unavailable models
python scripts/validate_models/validate_all.py --skip biomedparse medsam3
```

### 4. Run the Agent

```bash
# Single image
python scripts/run_agent.py --image /path/to/cxr.png

# Directory of images
python scripts/run_agent.py --image_dir /path/to/images/ --output results/

# Without CLEAR concept prior
python scripts/run_agent.py --image /path/to/cxr.png --no_clear

# Custom config
python scripts/run_agent.py --image /path/to/cxr.png --config configs/config.yaml
```

Output: JSON with report, trajectory (tool calls + reasoning), and token usage saved to `results/`.

## Configuration

`configs/config.yaml`:

```yaml
agent:
  model: "claude-sonnet-4-6"
  max_iterations: 10         # max ReAct loop iterations
  max_tokens: 4096           # per response (16000 when thinking is on)
  temperature: 0.0           # ignored when thinking is enabled
  reasoning_effort: "medium" # "low", "medium", "high", or null

clear:
  top_k: 20                  # concepts injected as prior

tools:
  chexagent2:
    enabled: true             # set false to disable specific tools
    endpoint: "http://localhost:8001"
  # ... (all 14 tools)
```

Disable tools by setting `enabled: false` in config. The agent adapts — it only sees tools that are enabled.

## Project Structure

```
CXR_Agent/
├── agent/
│   ├── react_agent.py       # ReAct loop, Anthropic API calls, adaptive thinking
│   └── prompts.py           # System prompt, concept prior template
├── clear/
│   ├── concept_scorer.py    # CLEAR model: CLIP + DinoV2 concept scoring
│   ├── clip_model.py        # CLIP architecture
│   └── clip_tokenizer.py    # CLIP text tokenizer
├── tools/
│   ├── base.py              # BaseCXRTool interface (name, description, schema, run)
│   ├── chexagent2.py        # 5 tool classes → :8001
│   ├── chexone.py           # 1 tool class  → :8002
│   ├── medversa.py          # 5 tool classes → :8004
│   ├── biomedparse.py       # 1 tool class  → :8005
│   ├── medsam3.py           # 1 tool class  → :8006
│   └── factchexcker.py      # 1 tool class  → :8007
├── servers/
│   ├── chexagent2_server.py # Multi-task FastAPI (report, srrg, classify, ground, vqa)
│   ├── chexone_server.py    # FastAPI (report with optional reasoning)
│   ├── medversa_server.py   # Multi-task FastAPI (report, classify, detect, segment, vqa)
│   ├── biomedparse_server.py
│   ├── medsam3_server.py
│   └── factchexcker_server.py
├── skills/                  # Empty — reserved for evotest evolved skills
├── configs/
│   └── config.yaml
├── scripts/
│   ├── run_agent.py         # Main entry point
│   ├── launch_servers.sh    # Launch/stop all model servers
│   ├── precompute_concepts.py
│   └── validate_models/     # Per-model validation scripts
└── tmp/                     # Reference code from cxr_concept and mimic_skills
```

## GPU Allocation (3x A6000)

| GPU | Server | Models | VRAM Est. |
|-----|--------|--------|-----------|
| 0 | :8001 | CheXagent-2 (3B, multi-task) + CheXagent-2-SRRG (shared) | ~8 GB |
| 1 | :8002, :8005 | CheXOne (3B) + BiomedParse | ~10 GB |
| 2 | :8004, :8006, :8007 | MedVersa (7B) + MedSAM3 + FactCheXcker | ~20 GB |

CLEAR runs in-process on whichever GPU the agent process uses (CPU also works, ~2 GB).

## Next Steps

### GPU server setup — March 11

1. **Clone external repos** (MedVersa, BiomedParse, MedSAM3, FactCheXcker) — see `scripts/validate_models/GPU_SERVER_SETUP.md`
2. **Install environments** — base conda env + per-model deps from `envs/*.txt`
3. **Download model weights** — HuggingFace models auto-download on first use; CLEAR checkpoint is manual
4. **Validate each model** — `python scripts/validate_models/validate_all.py`
5. **Launch all 6 servers** — `bash scripts/launch_servers.sh` and verify health endpoints
6. **Smoke test the agent** — `python scripts/run_agent.py --image /path/to/test_cxr.png`

### MIMIC-CXR evaluation: CheXOne (baseline) vs. CXR Agent — March 12

Goal: compare reports from CheXOne alone vs. full agent on the MIMIC-CXR test set using ReXrank metrics.

1. **Prepare MIMIC-CXR test split** — extract test image paths + ground truth reports (FINDINGS + IMPRESSION) into a JSON
2. **Run CheXOne baseline** — for each test image, call CheXOne server directly (no agent, no CLEAR prior), save generated reports
3. **Run CXR Agent** — for each test image, run the full agent pipeline (`run_agent.py`), save generated reports
4. **Install [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)** — computes all ReXrank metrics
5. **Score both** — compute RadCliQ-v1 (primary), RadGraph-F1, SembScore, BERTScore, BLEU-2 for baseline and agent
6. **Compare** — side-by-side metric table, per-study analysis of where the agent helps vs. hurts

Deliverable: `scripts/eval_mimic.py` with `--mode chexone` (baseline) and `--mode agent` (ours).

### Future

- **Prior + current CXR comparison**: Accept a prior CXR alongside the current study, compute CLEAR priors for both, pass both images to multi-image-capable tools (CheXagent-2, CheXOne), lift the no-comparison constraint to report interval changes.
- **Evotest skill evolution**: The `skills/` directory and skill injection infrastructure are in place. The fixed `SYSTEM_PROMPT` contains only hard constraints. All clinical reasoning strategy is meant to be evolved via reward-driven optimization.
- **CRIMSON / ReXrank reward wrappers**: Reward models for evolutionary skill optimization.
- **Interactive reporting frontend**: Click on findings to highlight segmentation masks on the CXR.
- **MedVersa task string verification**: Exact task parameter values need empirical testing against the MedVersa repo.
- **ReXrank leaderboard submission**: After MIMIC-CXR eval, prepare inference script in ReXrank format (`python inference.py <input_json> <output_json> <img_root>`) and submit to ReXrank for evaluation on the private ReXGradient dataset (10K studies, 67 sites).
