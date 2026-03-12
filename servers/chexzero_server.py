"""
CheXzero zero-shot classification FastAPI server.

Official ensemble approach from CheXzero paper:
    1. Load all 10 checkpoints
    2. For each model: compute softmax(pos_sim, neg_sim) per pathology
    3. Average predictions across all models

Templates: ("{}", "no {}") — exactly as in the paper.

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/chexzero_server.py --port 8009 \
        --chexzero_dir /path/to/CheXzero \
        --model_dir /path/to/CheXzero/checkpoints/CheXzero_Models
"""

import argparse
import glob
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from PIL import Image

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("chexzero_server")

state = {}

# Default 14 CheXpert pathology labels
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Positive/negative template pair (from CheXzero paper)
POS_TEMPLATE = "{}"
NEG_TEMPLATE = "no {}"


class ClassifyRequest(BaseModel):
    image_path: str
    pathologies: Optional[List[str]] = None  # defaults to CHEXPERT_LABELS


class ClassifyResponse(BaseModel):
    predictions: dict  # pathology -> probability [0,1]
    inference_time_ms: float
    num_models: int


def _preprocess_image(image_path: str, desired_size: int = 320) -> torch.Tensor:
    """Load and preprocess CXR image matching CheXzero's original pipeline.

    From CheXzero data_process.py preprocess():
    1. Load grayscale
    2. Aspect-ratio-preserving resize so max dim = desired_size
    3. Zero-pad to desired_size × desired_size square
    Then from zero_shot.py make() for pretrained=True:
    4. Expand to 3 channels, normalize with CXR stats
    5. Resize to 224×224 (CLIP ViT-B/32 input resolution)
    """
    img = Image.open(image_path).convert("L")  # grayscale
    old_size = img.size  # (W, H)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    padded = Image.new("L", (desired_size, desired_size))
    padded.paste(img, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))
    arr = np.array(padded, dtype=np.float32)
    arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
    tensor = torch.from_numpy(arr)
    tensor = state["transform"](tensor)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def _compute_text_weights(model, tokenize, device, pathologies, template):
    """Compute normalized text embeddings for a list of pathologies."""
    with torch.no_grad():
        weights = []
        for name in pathologies:
            text = template.format(name)
            tokens = tokenize([text], context_length=77).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            weights.append(emb.squeeze(0))
        return torch.stack(weights, dim=1)  # (embed_dim, num_classes)


def _classify_single_model(model_idx, image_tensor, pathologies):
    """Run softmax eval on a single image with one model (official CheXzero approach)."""
    model = state["models"][model_idx]
    device = state["device"]

    cache_key = (model_idx, tuple(pathologies))
    if cache_key not in state["text_cache"]:
        tokenize = state["tokenize"]
        pos_w = _compute_text_weights(model, tokenize, device, pathologies, POS_TEMPLATE)
        neg_w = _compute_text_weights(model, tokenize, device, pathologies, NEG_TEMPLATE)
        state["text_cache"][cache_key] = (pos_w, neg_w)

    pos_weights, neg_weights = state["text_cache"][cache_key]

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        img_features = model.encode_image(image_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Apply learned logit_scale (temperature) for meaningful separation,
        # then softmax to get probabilities.
        logit_scale = model.logit_scale.exp()
        pos_logits = (logit_scale * img_features @ pos_weights).squeeze(0).cpu().numpy()
        neg_logits = (logit_scale * img_features @ neg_weights).squeeze(0).cpu().numpy()

        max_logits = np.maximum(pos_logits, neg_logits)
        exp_pos = np.exp(pos_logits - max_logits)
        exp_neg = np.exp(neg_logits - max_logits)
        probs = exp_pos / (exp_pos + exp_neg)

    return probs  # (num_classes,)


def _classify_ensemble(image_tensor, pathologies):
    """Ensemble: average probabilities across 10 models, then threshold → present/absent."""
    all_probs = []
    for i in range(len(state["models"])):
        probs = _classify_single_model(i, image_tensor, pathologies)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    results = {}
    for name, prob in zip(pathologies, avg_probs):
        results[name] = "present" if prob > 0.5 else "absent"
    return results


def _build_model(chexzero_dir, device):
    """Build a single CheXzero CLIP model (ViT-B/32 architecture)."""
    from model import CLIP, VisualTransformer, Transformer, LayerNorm

    vision_heads = 768 // 64  # 12
    visual = VisualTransformer(
        input_resolution=224, patch_size=32, width=768,
        layers=12, heads=vision_heads, output_dim=512,
    )

    class CheXzeroCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length = 77
            self.visual = visual
            self.transformer = Transformer(
                width=512, layers=12, heads=8,
                attn_mask=self._build_attention_mask()
            )
            self.token_embedding = torch.nn.Embedding(49408, 512)
            self.positional_embedding = torch.nn.Parameter(torch.empty(77, 512))
            self.ln_final = LayerNorm(512)
            self.text_projection = torch.nn.Parameter(torch.empty(512, 512))
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.initialize_parameters()

        def _build_attention_mask(self):
            mask = torch.empty(77, 77)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            return mask

        def initialize_parameters(self):
            torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
            torch.nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (512 ** -0.5) * ((2 * 12) ** -0.5)
            attn_std = 512 ** -0.5
            fc_std = (2 * 512) ** -0.5
            for block in self.transformer.resblocks:
                torch.nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                torch.nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                torch.nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                torch.nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            if self.text_projection is not None:
                torch.nn.init.normal_(self.text_projection, std=512 ** -0.5)

        def encode_image(self, image):
            return self.visual(image.float())

        def encode_text(self, text):
            x = self.token_embedding(text)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x

    return CheXzeroCLIP()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chexzero_dir = state.get("chexzero_dir", "")
    model_dir = state.get("model_dir", "")
    device = state.get("device", "cuda")

    sys.path.insert(0, chexzero_dir)
    from clip import tokenize as clip_tokenize

    # Find all checkpoints
    model_paths = sorted(glob.glob(f"{model_dir}/*.pt"))
    if not model_paths:
        raise RuntimeError(f"No .pt checkpoints found in {model_dir}")

    logger.info(f"Loading {len(model_paths)} CheXzero checkpoints for ensemble...")

    models = []
    for i, path in enumerate(model_paths):
        name = Path(path).stem
        logger.info(f"  [{i+1}/{len(model_paths)}] Loading {name}...")
        model = _build_model(chexzero_dir, device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        models.append(model)

    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])

    state["models"] = models
    state["tokenize"] = clip_tokenize
    state["transform"] = transform
    state["device"] = device
    state["text_cache"] = {}

    # Pre-compute text embeddings for default labels on all models
    logger.info("Pre-computing text embeddings for 14 CheXpert labels...")
    for i in range(len(models)):
        pos_w = _compute_text_weights(models[i], clip_tokenize, device, CHEXPERT_LABELS, POS_TEMPLATE)
        neg_w = _compute_text_weights(models[i], clip_tokenize, device, CHEXPERT_LABELS, NEG_TEMPLATE)
        state["text_cache"][(i, tuple(CHEXPERT_LABELS))] = (pos_w, neg_w)

    logger.info(f"CheXzero ensemble ready ({len(models)} models)")
    yield
    state.clear()


app = FastAPI(title="CheXzero Ensemble Server", lifespan=lifespan)


@app.get("/health")
async def health():
    n = len(state.get("models", []))
    return {"status": "ok", "model": f"CheXzero-ensemble-{n}", "labels": len(CHEXPERT_LABELS)}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    pathologies = req.pathologies or CHEXPERT_LABELS
    t0 = time.time()
    image_tensor = _preprocess_image(req.image_path)
    preds = _classify_ensemble(image_tensor, pathologies)
    inference_time = (time.time() - t0) * 1000
    return ClassifyResponse(
        predictions=preds,
        inference_time_ms=inference_time,
        num_models=len(state["models"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--chexzero_dir", type=str, default="/home/than/DeepLearning/CheXzero")
    parser.add_argument("--model_dir", type=str,
                        default="/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    state["chexzero_dir"] = args.chexzero_dir
    state["model_dir"] = args.model_dir
    state["device"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
