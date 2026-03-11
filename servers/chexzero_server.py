"""
CheXzero zero-shot classification FastAPI server.

Uses positive/negative prompt pairs and softmax between them:
    P(finding) = exp(sim(img, "Atelectasis")) / (exp(sim(img, "Atelectasis")) + exp(sim(img, "no Atelectasis")))

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/chexzero_server.py --port 8008 \
        --chexzero_dir /path/to/CheXzero
"""

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager
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
    # Aspect-ratio-preserving resize + zero-pad (matches data_process.py)
    old_size = img.size  # (W, H)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    padded = Image.new("L", (desired_size, desired_size))
    padded.paste(img, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))
    # Convert to 3-channel float32 tensor
    arr = np.array(padded, dtype=np.float32)
    arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
    tensor = torch.from_numpy(arr)
    # Apply CheXzero normalization + resize to 224
    tensor = state["transform"](tensor)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def _compute_text_weights(pathologies: List[str], template: str) -> torch.Tensor:
    """Compute text embeddings for a list of pathologies with given template."""
    model = state["model"]
    tokenize = state["tokenize"]

    with torch.no_grad():
        weights = []
        for name in pathologies:
            text = template.format(name)
            tokens = tokenize([text], context_length=77)
            tokens = tokens.to(state["device"])
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            weights.append(emb.squeeze(0))
        return torch.stack(weights, dim=1)  # (embed_dim, num_classes)


def _classify_single(image_tensor: torch.Tensor, pathologies: List[str]) -> dict:
    """Run softmax eval on a single image for given pathologies."""
    model = state["model"]
    device = state["device"]

    # Check if we have cached weights for these exact pathologies
    cache_key = tuple(pathologies)
    if cache_key not in state.get("text_cache", {}):
        pos_weights = _compute_text_weights(pathologies, POS_TEMPLATE)
        neg_weights = _compute_text_weights(pathologies, NEG_TEMPLATE)
        if "text_cache" not in state:
            state["text_cache"] = {}
        state["text_cache"][cache_key] = (pos_weights, neg_weights)

    pos_weights, neg_weights = state["text_cache"][cache_key]

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        img_features = model.encode_image(image_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Cosine similarity to positive and negative
        pos_logits = (img_features @ pos_weights).squeeze(0).cpu().numpy()  # (num_classes,)
        neg_logits = (img_features @ neg_weights).squeeze(0).cpu().numpy()

        # Softmax between positive and negative
        sum_exp = np.exp(pos_logits) + np.exp(neg_logits)
        probs = np.exp(pos_logits) / sum_exp

    return {name: round(float(p), 4) for name, p in zip(pathologies, probs)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chexzero_dir = state.get("chexzero_dir", "")
    model_path = state.get("model_path", "")
    device = state.get("device", "cuda")

    # Add CheXzero to path for imports
    sys.path.insert(0, chexzero_dir)

    # Import CLIP model class (use original VisualTransformer, not DinoV2)
    from model import CLIP, VisualTransformer
    from clip import tokenize as clip_tokenize

    logger.info(f"Loading CheXzero model from {model_path}...")

    # Build model with original ViT-B/32 architecture
    # These params match the released checkpoint
    params = {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 32,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    }

    # Temporarily restore VisualTransformer in CLIP if model.py was patched
    # We need the original VisualTransformer, not DinoV2
    import model as model_module
    original_clip_init = CLIP.__init__

    def patched_init(self, embed_dim, image_resolution, vision_layers, vision_width,
                     vision_patch_size, context_length, vocab_size, transformer_width,
                     transformer_heads, transformer_layers):
        nn_Module = torch.nn.Module
        nn_Module.__init__(self)
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            # ResNet path (not used for our checkpoint)
            from model import ModifiedResNet
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers, output_dim=embed_dim,
                heads=vision_heads, input_resolution=image_resolution, width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        from model import Transformer, LayerNorm
        self.transformer = Transformer(
            width=transformer_width, layers=transformer_layers,
            heads=transformer_heads, attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = torch.nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = torch.nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = torch.nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    CLIP.__init__ = patched_init
    model = CLIP(**params)
    CLIP.__init__ = original_clip_init  # restore

    # Load weights
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    # Image transform: CXR normalization → resize to 224×224 (CLIP ViT-B/32 resolution)
    # Original CheXzero uses already-square 320×320 h5 images, so Resize(224) works.
    # Here images may be non-square, so we force (224, 224) for correct patch grid.
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])

    state["model"] = model
    state["tokenize"] = clip_tokenize
    state["transform"] = transform
    state["device"] = device

    # Pre-compute text embeddings for default 14 CheXpert labels
    logger.info("Pre-computing text embeddings for 14 CheXpert labels...")
    pos_w = _compute_text_weights(CHEXPERT_LABELS, POS_TEMPLATE)
    neg_w = _compute_text_weights(CHEXPERT_LABELS, NEG_TEMPLATE)
    state["text_cache"] = {tuple(CHEXPERT_LABELS): (pos_w, neg_w)}

    logger.info("CheXzero ready")
    yield
    state.clear()


app = FastAPI(title="CheXzero Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "CheXzero", "labels": len(CHEXPERT_LABELS)}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    pathologies = req.pathologies or CHEXPERT_LABELS
    t0 = time.time()
    image_tensor = _preprocess_image(req.image_path)
    preds = _classify_single(image_tensor, pathologies)
    inference_time = (time.time() - t0) * 1000
    return ClassifyResponse(predictions=preds, inference_time_ms=inference_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--chexzero_dir", type=str, default="/home/than/DeepLearning/CheXzero")
    parser.add_argument("--model_path", type=str,
                        default="/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    state["chexzero_dir"] = args.chexzero_dir
    state["model_path"] = args.model_path
    state["device"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
