# GPU Server Setup Guide (3x A6000)

## GPU Allocation Plan

| GPU | Models | VRAM Est. |
|-----|--------|-----------|
| GPU 0 | CheXagent-2 (multi-task, 3B, ~6GB) + CheXagent-2-SRRG (same arch) + CLEAR (~2GB) | ~8GB |
| GPU 1 | CheXOne (3B, ~6GB) + BiomedParse (~4GB) | ~10GB |
| GPU 2 | MedVersa (multi-task, 7B, ~14GB) + MedSAM3 (~4GB) + FactCheXcker (~2GB) | ~20GB |

## Step 1: Base Environment

```bash
conda create -n cxr_agent python=3.10 -y
conda activate cxr_agent

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install anthropic tenacity pyyaml numpy h5py Pillow ftfy regex
pip install transformers accelerate sentencepiece protobuf
pip install fastapi uvicorn requests

# For CheXOne (Qwen2.5-VL)
pip install qwen-vl-utils

# For FactCheXcker
pip install factchexcker-carinanet
```

## Step 2: Clone External Repos

```bash
cd /path/to/your/repos  # parent of CXR_Agent

# MedVersa
git clone https://huggingface.co/hyzhou/MedVersa
cd MedVersa && conda env create -f environment.yml && cd ..

# BiomedParse
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse
pip install -r assets/requirements/requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
cd ..

# MedSAM3
git clone https://github.com/Joey-S-Liu/MedSAM3.git
cd MedSAM3
pip install -e .
cd ..

# FactCheXcker
git clone https://github.com/rajpurkarlab/FactCheXcker.git
cd FactCheXcker
pip install -r requirements.txt
cd ..
```

## Step 3: Download Model Weights

HuggingFace models (auto-downloaded on first use):
- `StanfordAIMI/CheXagent-2-3b` (~6GB)
- `StanfordAIMI/CheXagent-2-3b-srrg-findings` (~6GB)
- `StanfordAIMI/CheXOne` (~6GB)
- `hyzhou/MedVersa` (~14GB)
- `microsoft/BiomedParse` (auto via hf_hub)

Manual downloads:
- CLEAR: checkpoint at `../cxr_concept/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt`

## Step 4: Validate

```bash
cd CXR_Agent

# Run all validations (downloads a sample CXR automatically)
python scripts/validate_models/validate_all.py

# Or test individual models
python scripts/validate_models/validate_chexagent2.py --image /path/to/cxr.jpg
python scripts/validate_models/validate_chexone.py --image /path/to/cxr.jpg
python scripts/validate_models/validate_medversa.py --image /path/to/cxr.jpg
python scripts/validate_models/validate_clear.py --image /path/to/cxr.jpg
python scripts/validate_models/validate_biomedparse.py --image /path/to/cxr.jpg

# Skip models not yet set up
python scripts/validate_models/validate_all.py --skip biomedparse medsam3 factchexcker

# Run only specific models
python scripts/validate_models/validate_all.py --only chexagent2 chexone clear
```

## Step 5: Launch Servers

```bash
# Launch all servers with GPU assignment
bash scripts/launch_servers.sh

# Or launch core servers only (CheXagent-2 + CheXOne)
bash scripts/launch_servers.sh --only core

# Stop all servers
bash scripts/launch_servers.sh --stop
```

## Troubleshooting

**CheXagent-2 `trust_remote_code` errors**: Make sure `transformers>=4.40`. The model uses custom tokenizer code.

**CheXOne missing `qwen_vl_utils`**: Install with `pip install qwen-vl-utils`.

**BiomedParse detectron2 errors**: Must install detectron2 from source: `pip install git+https://github.com/facebookresearch/detectron2.git`

**MedVersa import errors**: MedVersa needs its own conda environment. Activate it before launching the MedVersa server.

**CLEAR model not found**: Ensure `../cxr_concept/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt` exists. Or pass `--model_path` explicitly.
