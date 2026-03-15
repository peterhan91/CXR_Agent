"""
Google CXR Foundation zero-shot classification FastAPI server.

Replaces CheXzero with Google's CXR Foundation (ELIXR v2.0):
    - EfficientNet-L2 vision encoder + BERT text encoder
    - Contrastive embeddings for zero-shot classification
    - Mean AUC 0.846 across 13 CheXpert findings

Uses positive/negative prompt pairs and softmax between them:
    P(finding) = softmax(sim(img, "finding present"), sim(img, "no finding"))

Same API contract as chexzero_server.py — drop-in replacement on port 8008.

Usage:
    CUDA_VISIBLE_DEVICES=1 python servers/cxr_foundation_server.py --port 8008
"""

import argparse
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import png
from PIL import Image


def _load_cxr(image_path: str, mode: str = "L") -> Image.Image:
    """Load CXR image, properly normalizing 16-bit PNGs to 8-bit.

    PIL's .convert() silently clips 16-bit (mode I) images to 8-bit,
    destroying the dynamic range. PadChest-GR and RexGradient use 16-bit PNGs.
    """
    img = Image.open(image_path)
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
        img = img.convert("L")
    if mode == "RGB":
        img = img.convert("RGB")
    return img

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("cxr_foundation_server")

state = {}

# Default 14 CheXpert pathology labels (same as CheXzero for compatibility)
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Same prompt templates as CheXzero: "{finding}" vs "no {finding}"
POS_TEMPLATE = "{}"
NEG_TEMPLATE = "no {}"


class ClassifyRequest(BaseModel):
    image_path: str
    pathologies: Optional[List[str]] = None


class ClassifyResponse(BaseModel):
    predictions: dict
    inference_time_ms: float


def _image_to_tf_example(image_path: str):
    """Load CXR image and convert to tf.train.Example (CXR Foundation format).

    Pipeline from Google's data_processing_lib.py:
    1. Load as grayscale
    2. Convert to uint16, rescale dynamic range
    3. Encode as PNG
    4. Wrap in tf.train.Example with 'image/encoded' feature
    """
    import tensorflow as tf

    img = _load_cxr(image_path, mode="L")  # 16-bit safe
    arr = np.array(img)

    # Convert to uint16 and rescale to full range (matches image_utils.py)
    pixel_array = arr.astype(np.float64)
    pixel_array = pixel_array - pixel_array.min()
    max_val = pixel_array.max()
    if max_val > 0:
        pixel_array = pixel_array * (65535.0 / max_val)
    pixel_array = pixel_array.astype(np.uint16)

    # Encode as PNG (matches image_utils.encode_png)
    writer = png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=16,
    )
    output = io.BytesIO()
    writer.write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create tf.train.Example
    example = tf.train.Example()
    features = example.features.feature
    features["image/encoded"].bytes_list.value.append(png_bytes)
    features["image/format"].bytes_list.value.append(b"png")
    return example


def _bert_tokenize(text: str):
    """Tokenize text using BERT preprocessor (matches CXR Foundation's pipeline)."""
    import tensorflow as tf

    bert_model = state["bert_model"]
    out = bert_model(tf.constant([text.lower()]))
    ids = out["input_word_ids"].numpy().astype(np.int32)
    masks = out["input_mask"].numpy().astype(np.float32)
    paddings = 1.0 - masks

    # Remove SEP token (id=102) — CXR Foundation convention
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0

    ids = np.expand_dims(ids, axis=1)       # (1, 1, 128)
    paddings = np.expand_dims(paddings, axis=1)  # (1, 1, 128)
    return ids, paddings


def _get_text_embedding(text: str) -> np.ndarray:
    """Get contrastive text embedding for a single text query."""
    import tensorflow as tf

    ids, paddings = _bert_tokenize(text)
    qformer = state["qformer_model"]
    output = qformer.signatures["serving_default"](
        ids=tf.constant(ids),
        paddings=tf.constant(paddings),
        image_feature=np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
    )
    return output["contrastive_txt_emb"][0].numpy()  # shape (128,)


def _get_image_embedding(image_path: str) -> np.ndarray:
    """Get contrastive image embedding for a CXR image."""
    import tensorflow as tf

    example = _image_to_tf_example(image_path)
    elixr_c = state["elixr_c_model"]
    elixr_c_output = elixr_c.signatures["serving_default"](
        input_example=tf.constant([example.SerializeToString()])
    )
    image_features = elixr_c_output["feature_maps_0"]

    qformer = state["qformer_model"]
    qformer_output = qformer.signatures["serving_default"](
        image_feature=image_features.numpy().tolist(),
        ids=np.zeros([1, 1, 128], dtype=np.int32).tolist(),
        paddings=np.zeros([1, 1, 128], dtype=np.float32).tolist(),
    )
    # Contrastive image embedding: (32, 128)
    return qformer_output["all_contrastive_img_emb"][0].numpy()


def _compute_image_text_similarity(image_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    """Cosine similarity between image (32×128) and text (128,) embeddings.

    Takes max over 32 image patches (matches CXR Foundation's approach).
    """
    image_emb = image_emb.reshape(32, 128)
    similarities = []
    for i in range(32):
        norm_img = np.linalg.norm(image_emb[i])
        norm_txt = np.linalg.norm(txt_emb)
        if norm_img > 0 and norm_txt > 0:
            sim = np.dot(image_emb[i], txt_emb) / (norm_img * norm_txt)
        else:
            sim = 0.0
        similarities.append(sim)
    return float(np.max(similarities))


def _classify_single(image_path: str, pathologies: List[str], debug: bool = False) -> dict:
    """Zero-shot classification for a single image."""
    image_emb = _get_image_embedding(image_path)

    results = {}
    debug_info = {}
    for name in pathologies:
        cache_key = name
        if cache_key not in state.get("text_cache", {}):
            pos_text = POS_TEMPLATE.format(name)
            neg_text = NEG_TEMPLATE.format(name)
            pos_emb = _get_text_embedding(pos_text)
            neg_emb = _get_text_embedding(neg_text)
            if "text_cache" not in state:
                state["text_cache"] = {}
            state["text_cache"][cache_key] = (pos_emb, neg_emb)

        pos_emb, neg_emb = state["text_cache"][cache_key]
        pos_sim = _compute_image_text_similarity(image_emb, pos_emb)
        neg_sim = _compute_image_text_similarity(image_emb, neg_emb)

        diff = pos_sim - neg_sim
        # Require minimum margin to counter systematic positive biases
        # (e.g., "Fracture" has mean diff +0.035 even on normal CXRs)
        # Calibrated on 25 val images: margin=0.08 gives ~1.0 positives/study
        MARGIN = 0.08
        results[name] = "present" if diff > MARGIN else "absent"
        if debug:
            debug_info[name] = {"pos_sim": round(pos_sim, 4), "neg_sim": round(neg_sim, 4), "diff": round(diff, 4)}

    if debug:
        return results, debug_info
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    import tensorflow as tf
    import tensorflow_hub as tf_hub
    import tensorflow_text  # noqa: F401 — registers SentencepieceOp before model load

    model_dir = state.get("model_dir", "models/cxr-foundation")
    device = state.get("device", "gpu")

    # GPU memory config
    if device.startswith("gpu"):
        gpu_id = 0
        if ":" in device:
            gpu_id = int(device.split(":")[1])
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            target_gpu = gpus[min(gpu_id, len(gpus) - 1)]
            tf.config.set_visible_devices([target_gpu], "GPU")
            tf.config.experimental.set_memory_growth(target_gpu, True)
            logger.info(f"Using GPU: {target_gpu}")
        else:
            logger.warning("No GPUs found, falling back to CPU")
    elif device == "cpu":
        tf.config.set_visible_devices([], "GPU")

    logger.info(f"Loading ELIXR-C model from {model_dir}/elixr-c-v2-pooled...")
    elixr_c_model = tf.saved_model.load(f"{model_dir}/elixr-c-v2-pooled")
    logger.info("ELIXR-C loaded")

    logger.info(f"Loading QFormer model from {model_dir}/pax-elixr-b-text...")
    qformer_model = tf.saved_model.load(f"{model_dir}/pax-elixr-b-text")
    logger.info("QFormer loaded")

    logger.info("Loading BERT tokenizer from TF Hub...")
    bert_model = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    logger.info("BERT tokenizer loaded")

    state["elixr_c_model"] = elixr_c_model
    state["qformer_model"] = qformer_model
    state["bert_model"] = bert_model

    # Pre-compute text embeddings for all 14 CheXpert labels
    logger.info("Pre-computing text embeddings for 14 CheXpert labels...")
    state["text_cache"] = {}
    for name in CHEXPERT_LABELS:
        pos_text = POS_TEMPLATE.format(name)
        neg_text = NEG_TEMPLATE.format(name)
        pos_emb = _get_text_embedding(pos_text)
        neg_emb = _get_text_embedding(neg_text)
        state["text_cache"][name] = (pos_emb, neg_emb)
    logger.info("CXR Foundation ready")

    yield
    state.clear()


app = FastAPI(title="CXR Foundation Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "CXR-Foundation-v2", "labels": len(CHEXPERT_LABELS)}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    pathologies = req.pathologies or CHEXPERT_LABELS
    t0 = time.time()
    preds = _classify_single(req.image_path, pathologies)
    inference_time = (time.time() - t0) * 1000
    return ClassifyResponse(predictions=preds, inference_time_ms=inference_time)


@app.post("/classify_debug")
async def classify_debug(req: ClassifyRequest):
    """Debug endpoint: returns raw cosine similarities and diffs."""
    pathologies = req.pathologies or CHEXPERT_LABELS
    t0 = time.time()
    preds, debug_info = _classify_single(req.image_path, pathologies, debug=True)
    inference_time = (time.time() - t0) * 1000
    return {"predictions": preds, "debug": debug_info, "inference_time_ms": inference_time}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_dir", type=str,
                        default="/home/than/DeepLearning/CXR_Agent/models/cxr-foundation")
    parser.add_argument("--device", type=str, default="gpu:1",
                        help="Device: gpu:0, gpu:1, gpu:2, or cpu")
    args = parser.parse_args()

    state["model_dir"] = args.model_dir
    state["device"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
