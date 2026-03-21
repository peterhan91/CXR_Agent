# CXR Agent — Full Setup Guide (DGX Spark)

Complete setup instructions to replicate the CXR Agent + Web UI on a fresh machine.
Written for Claude Code to execute autonomously — run `claude` in the repo root and
paste this file or say "follow SETUP.md".

## Prerequisites

- Ubuntu 22.04+ with NVIDIA GPUs (3x 24GB+ or 2x 48GB+)
- NVIDIA driver 535+ with CUDA 12.x
- Conda (Miniconda or Anaconda)
- Node.js 20+ and npm 10+
- Git
- ~80GB free disk (models auto-download from HuggingFace)

## Architecture Overview

```
Browser (:3000) → Next.js UI → FastAPI Gateway (:9000) → CXRReActAgent (Claude Sonnet 4.6)
                                                              ↓ tool calls
                                                         Model Servers (:8001-:8011)
```

**Config used:** `configs/config_combo_full.yaml` (10 enabled tools, 5 servers)

### Servers needed for config_combo_full

| GPU | Server | Port | Conda Env | Tools |
|-----|--------|------|-----------|-------|
| 0 | CheXagent-2 | 8001 | `cxr_chexagent2` | report, srrg, classify, vqa, temporal |
| 1 | CheXOne | 8002 | `cxr_agent` | report |
| 1 | BiomedParse | 8005 | `cxr_agent` | segment |
| 2 | MedGemma | 8010 | `cxr_agent` | vqa, grounding |
| 2 | FactCheXcker | 8007 | `cxr_agent` | verify |

**Not needed** (disabled in config_combo_full): CheXzero (:8009), CXR Foundation (:8008), MedVersa (:8004), Whisper (:8011)

---

## Step 0: Clone the repo

```bash
cd ~/DeepLearning
git clone https://github.com/peterhan91/CXR_Agent.git
cd CXR_Agent
```

## Step 1: API keys

Create a `.env` file (or export in your shell):

```bash
# Required — the agent brain
export ANTHROPIC_API_KEY="sk-ant-..."

# Required — FactCheXcker verification pipeline
export OPENAI_API_KEY="sk-..."

# Optional — only if using MedVersa (disabled)
# export HF_TOKEN="hf_..."
```

## Step 2: Download dataset (eval JSONs + CXR images)

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download the dataset (12.3 GB — eval JSONs with relative paths + 13,909 CXR images)
huggingface-cli download peterhan91/CXR_Agent_Data \
    --repo-type dataset \
    --local-dir data/cxr_agent_dataset

# The eval JSONs use relative paths like "images/mimic/p12/p12810135/s50981777/xxx.jpg"
# The gateway needs these to be absolute. We generate a symlink layout:
python scripts/setup_data.py
```

> `scripts/setup_data.py` is created in Step 7 below. It copies eval JSONs from the
> downloaded dataset into `data/eval/`, rewriting relative image paths to absolute paths
> under `data/cxr_agent_dataset/`.

## Step 3: Create conda environments

### 3a. Main environment (`cxr_agent`) — runs 4 of 5 servers + gateway + agent

```bash
conda create -n cxr_agent python=3.10 -y
conda activate cxr_agent

# PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Core agent deps
pip install anthropic>=0.84 tenacity pyyaml requests Pillow numpy

# FastAPI server deps
pip install fastapi uvicorn

# CheXOne (Qwen2.5-VL based)
pip install transformers>=4.57 accelerate qwen-vl-utils sentencepiece protobuf

# MedGemma
# (same deps as above — transformers>=4.57, accelerate)

# BiomedParse deps
pip install huggingface_hub hydra-core opencv-python
pip install git+https://github.com/facebookresearch/detectron2.git

# FactCheXcker
pip install carinanet openai

# CXR Foundation (TensorFlow — only if re-enabling cxr_foundation tool)
# pip install tensorflow tensorflow-hub tensorflow-text

# open_clip for CLEAR (only if re-enabling CLEAR)
# pip install open_clip_torch ftfy regex
```

### 3b. CheXagent-2 environment (`cxr_chexagent2`) — strict transformers==4.40.0

```bash
conda create -n cxr_chexagent2 python=3.10 -y
conda activate cxr_chexagent2

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate sentencepiece protobuf
pip install fastapi uvicorn Pillow numpy requests
pip install opencv-python albumentations einops
```

## Step 4: Clone external dependencies

```bash
cd ~/DeepLearning

# BiomedParse v1 (MUST use v1 branch — main/v2 is 3D only)
git clone --branch v1 https://github.com/microsoft/BiomedParse.git BiomedParse-v1
# If no v1 branch, clone main and checkout the v1 commit/tag

# FactCheXcker
git clone https://github.com/jbdel/FactCheXcker.git

cd CXR_Agent
```

## Step 5: Update hardcoded paths

The gateway has hardcoded allowed-image-roots. Update for your machine:

**`ui/server/gateway.py`** — update `ALLOWED_IMAGE_ROOTS`:
```python
ALLOWED_IMAGE_ROOTS = [
    # Point to wherever your dataset images live
    os.path.join(PROJECT_ROOT, "data"),
    os.path.join(PROJECT_ROOT, "results"),
]
```

**`servers/chexzero_server.py`** — only if re-enabling CheXzero (not needed for combo_full):
```python
parser.add_argument("--chexzero_dir", default="<YOUR_PATH>/CheXzero")
```

## Step 6: Install the Web UI

```bash
cd ui
npm install
cd ..
```

## Step 7: Create the data setup script

This script rewrites relative HF dataset paths to absolute paths for this machine.

Save as `scripts/setup_data.py`:

```python
#!/usr/bin/env python3
"""Rewrite HF dataset eval JSONs with absolute image paths for this machine."""
import json
import glob
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "cxr_agent_dataset"
EVAL_SRC = DATASET_DIR / "eval"
EVAL_DST = REPO_ROOT / "data" / "eval"

def rewrite_paths(obj, base_dir: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "path" in k.lower() and v.startswith("images/"):
                obj[k] = os.path.join(base_dir, v)
            elif isinstance(v, (dict, list)):
                rewrite_paths(v, base_dir)
    elif isinstance(obj, list):
        for item in obj:
            rewrite_paths(item, base_dir)

def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print("Run: huggingface-cli download peterhan91/CXR_Agent_Data --repo-type dataset --local-dir data/cxr_agent_dataset")
        return

    EVAL_DST.mkdir(parents=True, exist_ok=True)
    base = str(DATASET_DIR)

    for jf in sorted(glob.glob(str(EVAL_SRC / "**/*.json"), recursive=True)):
        rel = os.path.relpath(jf, EVAL_SRC)
        data = json.load(open(jf))
        rewrite_paths(data, base)
        out = EVAL_DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(out, "w"), indent=2)
        print(f"  Wrote {rel}")

    print(f"\nDone. Eval JSONs in {EVAL_DST} now use absolute paths under {DATASET_DIR}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python scripts/setup_data.py
```

## Step 8: Launch model servers

### Option A: Use the launch script (launches all in background)

Update `scripts/launch_servers.sh` if needed — the script auto-detects paths relative to repo root.

```bash
bash scripts/launch_servers.sh
```

### Option B: Launch only what config_combo_full needs (5 servers)

```bash
# Set these based on your conda install
CONDA_BASE="$(conda info --base)"
MAIN_PYTHON="${CONDA_BASE}/envs/cxr_agent/bin/python"
CA2_PYTHON="${CONDA_BASE}/envs/cxr_chexagent2/bin/python"
REPO_ROOT="$(pwd)"
PARENT_DIR="$(dirname $REPO_ROOT)"
mkdir -p logs

# GPU 0: CheXagent-2 (report, srrg, classify, vqa, temporal)
CUDA_VISIBLE_DEVICES=0 $CA2_PYTHON servers/chexagent2_server.py --port 8001 \
    > logs/chexagent2.log 2>&1 &

# GPU 1: CheXOne
CUDA_VISIBLE_DEVICES=1 $MAIN_PYTHON servers/chexone_server.py --port 8002 \
    > logs/chexone.log 2>&1 &

# GPU 1: BiomedParse
CUDA_VISIBLE_DEVICES=1 $MAIN_PYTHON servers/biomedparse_server.py --port 8005 \
    --biomedparse_dir "${PARENT_DIR}/BiomedParse-v1" \
    > logs/biomedparse.log 2>&1 &

# GPU 2: MedGemma
CUDA_VISIBLE_DEVICES=2 $MAIN_PYTHON servers/medgemma_server.py --port 8010 \
    > logs/medgemma.log 2>&1 &

# GPU 2: FactCheXcker
CUDA_VISIBLE_DEVICES=2 $MAIN_PYTHON servers/factchexcker_server.py --port 8007 \
    --factchexcker_dir "${PARENT_DIR}/FactCheXcker" \
    > logs/factchexcker.log 2>&1 &

echo "Servers launching... wait 2-3 min for models to load"
echo "Monitor: tail -f logs/*.log"
```

### Verify all servers are healthy

```bash
for port in 8001 8002 8005 8007 8010; do
    printf ":%s " $port
    curl -s http://localhost:$port/health | head -c 50
    echo
done
```

## Step 9: Start the gateway + UI

```bash
# Terminal 1: FastAPI gateway
conda activate cxr_agent
python ui/server/gateway.py --port 9000 &

# Terminal 2: Next.js frontend
cd ui && npm run dev
```

The UI is now at **http://localhost:3000**.

## Step 10: Verify everything works

1. Open http://localhost:3000 — you should see the study browser
2. Click any study → click "Run Agent"
3. Watch the trajectory stream in real-time
4. Final report appears with FINDINGS + IMPRESSION

### Health check endpoint
```bash
curl http://localhost:9000/api/servers/health
```

---

## GPU Memory Budget (config_combo_full)

| GPU | Server | VRAM |
|-----|--------|------|
| 0 | CheXagent-2 (3 model variants) | ~18 GB |
| 1 | CheXOne (Qwen2.5-VL-3B) + BiomedParse | ~18 GB |
| 2 | MedGemma (4B) + FactCheXcker (CarinaNet) | ~12 GB |

**Minimum**: 3x 24GB GPUs or 2x 48GB GPUs (rebalance: put CheXagent-2 + FactCheXcker on one, CheXOne + BiomedParse + MedGemma on the other).

## Troubleshooting

### BiomedParse "3D only" error
You cloned the wrong branch. BiomedParse main/v2 is 3D-only. You need the v1 branch or worktree.

### CheXagent-2 crashes with transformers error
CheXagent-2 requires `transformers==4.40.0`. Make sure it runs in `cxr_chexagent2` env, not `cxr_agent`.

### FactCheXcker "carinanet not found"
Run `pip install carinanet` in the `cxr_agent` env.

### Gateway returns 403 on images
Update `ALLOWED_IMAGE_ROOTS` in `ui/server/gateway.py` to include the directory where your CXR images are stored.

### HuggingFace models fail to download
Some models (CheXagent-2, CheXOne) may need HuggingFace login:
```bash
huggingface-cli login
```

### "No module named scripts.run_agent"
The gateway adds PROJECT_ROOT to sys.path. Make sure you start the gateway from the repo root:
```bash
cd /path/to/CXR_Agent
python ui/server/gateway.py --port 9000
```
