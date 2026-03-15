"""
CLEAR Concept Scorer for CXR images.

Adapted from cxr_concept/reports/01_compute_concept_scores.py.
Provides both:
- Single-image scoring (for agent inference)
- Batch scoring (for precomputing concept scores across a dataset)

The concept scoring pipeline:
1. Load CLEAR model (CLIP with DinoV2 vision encoder)
2. Load concept vocabulary (368K MIMIC-CXR observations)
3. Encode concepts with CLIP text encoder -> concept_features [N_concepts, 768]
4. Encode CXR image with CLIP vision encoder -> image_features [1, 768]
5. Compute cosine similarity: image_features @ concept_features.T -> [1, N_concepts]
6. Return top-K concepts with scores as structured prior for the agent
"""

import hashlib
import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cuda_device(device_str: str) -> bool:
    """Check if a device string refers to any CUDA device.

    Handles 'cuda', 'cuda:0', 'cuda:1', etc.
    Original cxr_concept code assumes CUDA unconditionally; we guard for CPU.
    """
    return str(device_str).startswith("cuda")

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent
_CXR_CONCEPT_ROOT = _PROJECT_ROOT.parent / "cxr_concept"

DEFAULT_MODEL_PATH = _CXR_CONCEPT_ROOT / "checkpoints" / "dinov2-multi-v1.0_vitb" / "best_model.pt"
DEFAULT_CONCEPTS_PATH = _CXR_CONCEPT_ROOT / "concepts" / "mimic_concepts.csv"


class CLEARConceptScorer:
    """
    CLEAR concept scoring for CXR images.

    Wraps the concept scoring pipeline from cxr_concept. Loads the CLIP model
    with DinoV2 vision encoder and computes cosine similarity between image
    embeddings and concept text embeddings.

    This serves as the "strong prior" for the CXR ReAct agent — before the agent
    calls any tools, it receives the top-K most relevant clinical observations
    for the given CXR image.

    Usage:
        scorer = CLEARConceptScorer(model_path="path/to/model.pt")
        scorer.load()

        # Single image
        prior_text = scorer.score_image("path/to/cxr.png", top_k=20)

        # Batch scoring
        all_scores = scorer.score_batch("path/to/images.h5")
    """

    def __init__(
        self,
        model_path: str = None,
        concepts_path: str = None,
        dinov2_model_name: str = "dinov2_vitb14",
        image_resolution: int = 448,
        context_length: int = 77,
        device: str = None,
    ):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.concepts_path = concepts_path or str(DEFAULT_CONCEPTS_PATH)
        self.dinov2_model_name = dinov2_model_name
        self.image_resolution = image_resolution
        self.context_length = context_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.concepts = None
        self.concept_categories = None
        self.concept_features = None

        # Caches: image embeddings (in-memory) + concept priors (disk-persistent)
        self._embedding_cache = {}   # image_path -> np.ndarray [768]
        self._prior_cache = {}       # (image_path, top_k) -> formatted text
        self._cache_dir = Path("cache/clear")

        # Image transform (matches cxr_concept training pipeline)
        self.transform = Compose([
            Normalize(
                (101.48761, 101.48761, 101.48761),
                (83.43944, 83.43944, 83.43944),
            ),
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
        ])

    def load(self):
        """Load the CLEAR model and encode all concepts.

        This is the expensive operation — call once, then score many images.
        """
        logger.info("Loading CLEAR model...")
        self.model = self._load_clip_model()
        self.model = self.model.to(self.device).eval()
        logger.info(f"CLEAR model loaded on {self.device}")

        logger.info("Loading and encoding concepts...")
        self.concepts, self.concept_categories = self._load_concepts()
        self.concept_features = self._encode_concepts(self.concepts)
        logger.info(f"Encoded {len(self.concepts)} concepts -> {self.concept_features.shape}")

    def _load_clip_model(self):
        """Load CLIP model with DinoV2 vision encoder.

        Adapted from cxr_concept/train.py:load_clip() and
        cxr_concept/zero_shot.py:load_clip().
        """
        from clear.clip_model import CLIP

        params = {
            "embed_dim": 768,
            "image_resolution": 320,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_patch_size": 16,
            "context_length": self.context_length,
            "vocab_size": 49408,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12,
        }

        model = CLIP(**params)

        # Replace visual encoder with DinoV2 (from cxr_concept/train.py)
        device = torch.device(self.device)
        dinov2_backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self.dinov2_model_name + "_reg",
            pretrained=True,
        )
        dinov2_backbone = dinov2_backbone.to(device)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            features = dinov2_backbone(dummy_input)
            backbone_dim = features.shape[-1]

        class DinoV2Visual(nn.Module):
            def __init__(self, backbone, backbone_dim, output_dim):
                super().__init__()
                self.backbone = backbone
                self.projection = nn.Linear(backbone_dim, output_dim)

            def forward(self, x):
                features = self.backbone(x)
                return self.projection(features)

            @property
            def conv1(self):
                return self.projection

        model.visual = DinoV2Visual(dinov2_backbone, backbone_dim, params["embed_dim"])

        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        return model

    def _load_concepts(self) -> tuple:
        """Load concepts from mimic_concepts.csv (368K MIMIC-CXR observations).

        Returns:
            (concepts_list, category_map) where category_map is empty
            (full concept set has no category labels).
        """
        import csv

        all_concepts = []
        with open(self.concepts_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_concepts.append(row["concept"])

        logger.info(f"Loaded {len(all_concepts)} concepts from {self.concepts_path}")
        return all_concepts, {}

    def _encode_concepts(self, concepts: list) -> torch.Tensor:
        """Encode concept texts using CLIP text encoder.

        Adapted from cxr_concept/reports/01_compute_concept_scores.py Step 3.
        """
        from clear.clip_tokenizer import tokenize

        batch_size = 512
        all_features = []

        with torch.no_grad():
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i : i + batch_size]
                tokens = tokenize(batch, context_length=self.context_length).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
                if _is_cuda_device(self.device):
                    torch.cuda.empty_cache()

        concept_features = torch.cat(all_features).to(self.device)
        return concept_features

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single CXR image.

        Supports common image formats (PNG, JPG, DICOM).
        Converts to the format expected by CLEAR: grayscale -> 3-channel -> normalized.
        """
        from PIL import Image

        img = Image.open(image_path)
        # Properly handle 16-bit PNGs (PadChest-GR, RexGradient)
        if img.mode in ("I", "I;16"):
            arr = np.array(img, dtype=np.float64)
            arr = arr - arr.min()
            mx = arr.max()
            if mx > 0:
                arr = (arr / mx * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")
        else:
            img = img.convert("L")  # grayscale

        # Resize to square (matching cxr_concept/data_process.py:preprocess)
        desired_size = 320
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.LANCZOS)

        # Zero-pad to square
        new_img = Image.new("L", (desired_size, desired_size))
        new_img.paste(
            img,
            ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2),
        )

        # Convert to tensor: [1, 320, 320] -> [3, 320, 320]
        img_array = np.array(new_img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).repeat(3, 1, 1)

        # Apply transforms (normalize + resize to 448)
        img_tensor = self.transform(img_tensor)

        return img_tensor.unsqueeze(0)  # [1, 3, 448, 448]

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get image embedding, using cache if available.

        Returns normalized [768] embedding vector. Caches in memory
        for within-session reuse and on disk for cross-session persistence.
        """
        # Check in-memory cache
        if image_path in self._embedding_cache:
            logger.debug(f"CLEAR embedding cache hit: {image_path}")
            return self._embedding_cache[image_path]

        # Check disk cache
        cache_key = hashlib.md5(image_path.encode()).hexdigest()
        disk_path = self._cache_dir / f"{cache_key}.npy"
        if disk_path.exists():
            embedding = np.load(disk_path)
            self._embedding_cache[image_path] = embedding
            logger.debug(f"CLEAR disk cache hit: {image_path}")
            return embedding

        # Compute fresh embedding
        img_tensor = self._preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            embedding = img_features.squeeze(0).cpu().numpy()

        # Store in both caches
        self._embedding_cache[image_path] = embedding
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(disk_path, embedding)

        return embedding

    def score_image(self, image_path: str, top_k: int = 20) -> str:
        """Score a single CXR image against all concepts.

        Returns a formatted text string suitable for injection into the
        agent's system prompt as a concept prior. Uses embedding cache.

        Args:
            image_path: Path to the CXR image
            top_k: Number of top concepts to include

        Returns:
            Formatted concept prior text for the agent
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Check prior text cache (exact match on image + top_k)
        cache_key = (image_path, top_k)
        if cache_key in self._prior_cache:
            logger.debug(f"CLEAR prior cache hit: {image_path} top_k={top_k}")
            return self._prior_cache[cache_key]

        embedding = self._get_image_embedding(image_path)
        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = (embedding_tensor @ self.concept_features.T).squeeze(0).cpu().numpy()

        result = self._format_concept_prior(scores, top_k)
        self._prior_cache[cache_key] = result
        return result

    def score_image_raw(self, image_path: str) -> np.ndarray:
        """Score a single image and return raw scores array.

        Returns:
            numpy array of shape [N_concepts] with cosine similarity scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        embedding = self._get_image_embedding(image_path)
        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = (embedding_tensor @ self.concept_features.T).squeeze(0).cpu().numpy()

        return scores

    def score_batch_h5(self, h5_path: str, batch_size: int = 32) -> np.ndarray:
        """Batch-score all images in an HDF5 file.

        Adapted from cxr_concept/reports/01_compute_concept_scores.py Step 4.

        Args:
            h5_path: Path to HDF5 file with 'cxr' dataset
            batch_size: Batch size for inference

        Returns:
            numpy array of shape [N_images, N_concepts]
        """
        import h5py
        from torch.utils.data import DataLoader

        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Reuse CXRTestDataset pattern from cxr_concept/zero_shot.py
        # Added explicit file handle management (original leaks the handle)
        class _H5Dataset(torch.utils.data.Dataset):
            def __init__(self, path, transform):
                self._h5_file = h5py.File(path, "r")
                self.dset = self._h5_file["cxr"]
                self.transform = transform

            def __len__(self):
                return len(self.dset)

            def __getitem__(self, idx):
                img = self.dset[idx]
                img = np.expand_dims(img, axis=0)
                img = np.repeat(img, 3, axis=0)
                img = torch.from_numpy(img).float()
                if self.transform:
                    img = self.transform(img)
                return img

            def close(self):
                self._h5_file.close()

        dataset = _H5Dataset(h5_path, self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        all_scores = []
        with torch.no_grad():
            for batch_idx, imgs in enumerate(loader):
                imgs = imgs.to(self.device)
                img_features = self.model.encode_image(imgs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                scores = (img_features @ self.concept_features.T).cpu().numpy()
                all_scores.append(scores)

                if batch_idx % 100 == 0 and _is_cuda_device(self.device):
                    torch.cuda.empty_cache()

        dataset.close()
        return np.concatenate(all_scores, axis=0)

    def _format_concept_prior(self, scores: np.ndarray, top_k: int) -> str:
        """Format concept scores into text for the agent's system prompt."""
        top_indices = np.argsort(scores)[::-1][:top_k]

        lines = []
        for rank, idx in enumerate(top_indices, 1):
            concept_text = self.concepts[idx]
            score = scores[idx]
            lines.append(f"  {rank:2d}. {concept_text} (score: {score:.3f})")

        from agent.prompts import CONCEPT_PRIOR_TEMPLATE
        return CONCEPT_PRIOR_TEMPLATE.format(
            num_concepts=len(self.concepts),
            concept_scores="\n".join(lines),
        )
