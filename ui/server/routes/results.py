"""
/api/results — serve eval results, trajectories, scores.
/api/studies — browse all 7,912 studies across 4 datasets (paginated).

Data loaded once at import from results/eval_v4/ and data/eval/.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "eval_v4"
DATA_DIR = PROJECT_ROOT / "data" / "eval"

# ── Load data at startup ─────────────────────────────────────────────────────

def _load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> dict:
    """Load JSONL keyed by study_id."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                out[record["study_id"]] = record
    return out


# ── Eval results (160 studies with predictions) ──────────────────────────────

_test_set: list[dict] = _load_json(RESULTS_DIR / "test_set.json")
_test_set_by_id: dict[str, dict] = {s["study_id"]: s for s in _test_set}

# Predictions by model
_predictions: dict[str, dict[str, dict]] = {}
for pred_file in RESULTS_DIR.glob("predictions_*.json"):
    model_name = pred_file.stem.replace("predictions_", "")
    preds = _load_json(pred_file)
    _predictions[model_name] = {p["study_id"]: p for p in preds}

# Trajectories
_trajectories: dict[str, dict[str, dict]] = {}
for traj_file in RESULTS_DIR.glob("trajectories_*.jsonl"):
    model_name = traj_file.stem.replace("trajectories_", "")
    _trajectories[model_name] = _load_jsonl(traj_file)

# Scores summary
_scores: list[dict] = []
scores_path = RESULTS_DIR / "scores" / "summary.json"
if scores_path.exists():
    _scores = _load_json(scores_path)

# IDs with predictions
_eval_study_ids: set[str] = set(_test_set_by_id.keys())

# Available models for comparison
_model_names = sorted(_predictions.keys())

# ── Full datasets (7,912 studies) ─────────────────────────────────────────────

_DATASET_FILES = {
    "mimic_cxr": "mimic_cxr_test.json",
    "iu_xray": "iu_xray_test.json",
    "chexpert_plus": "chexpert_plus_valid.json",
    "rexgradient": "rexgradient_test.json",
}

_all_studies_by_dataset: dict[str, list[dict]] = {}
_all_studies_by_id: dict[str, dict] = {}
_dataset_counts: dict[str, int] = {}

for ds_name, filename in _DATASET_FILES.items():
    ds_path = DATA_DIR / filename
    if ds_path.exists():
        entries = _load_json(ds_path)
        _all_studies_by_dataset[ds_name] = entries
        _dataset_counts[ds_name] = len(entries)
        for entry in entries:
            _all_studies_by_id[entry["study_id"]] = entry
    else:
        _all_studies_by_dataset[ds_name] = []
        _dataset_counts[ds_name] = 0


# ── Routes: browse all studies ────────────────────────────────────────────────

@router.get("/studies")
async def list_studies(
    dataset: str | None = None,
    page: int = 1,
    per_page: int = 50,
    search: str | None = None,
    has_results: bool | None = None,
):
    """Browse all studies across datasets, paginated."""
    if dataset and dataset in _all_studies_by_dataset:
        studies = _all_studies_by_dataset[dataset]
    else:
        studies = list(_all_studies_by_id.values())

    # Filter by search term
    if search:
        q = search.lower()
        studies = [s for s in studies if q in s["study_id"].lower()]

    # Filter by has results
    if has_results is True:
        studies = [s for s in studies if s["study_id"] in _eval_study_ids]
    elif has_results is False:
        studies = [s for s in studies if s["study_id"] not in _eval_study_ids]

    total = len(studies)
    start = (page - 1) * per_page
    end = start + per_page
    page_studies = studies[start:end]

    # Build summary for each study
    items = []
    for s in page_studies:
        sid = s["study_id"]
        has_eval = sid in _eval_study_ids
        agent_pred = _predictions.get("agent_initial", {}).get(sid, {})
        items.append({
            "study_id": sid,
            "dataset": s.get("dataset", ""),
            "image_path": s.get("image_path", ""),
            "is_followup": s.get("is_followup", False),
            "has_results": has_eval,
            "num_steps": agent_pred.get("num_steps") if has_eval else None,
            "wall_time_ms": agent_pred.get("wall_time_ms") if has_eval else None,
            "has_ritl": sid in _predictions.get("agent_initial_ritl_rerun", {}),
        })

    return {
        "studies": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "datasets": _dataset_counts,
    }


# ── Routes: eval results (pre-computed) ──────────────────────────────────────

@router.get("/results")
async def list_results(dataset: str | None = None, ritl_only: bool = False):
    """List eval_v4 studies with summary info."""
    studies = []
    for ts in _test_set:
        sid = ts["study_id"]
        agent_pred = _predictions.get("agent_initial", {}).get(sid, {})
        study = {
            "study_id": sid,
            "dataset": ts.get("dataset", ""),
            "image_path": ts.get("image_path", ""),
            "view_position": ts.get("metadata", {}).get("view_position", ""),
            "is_followup": ts.get("is_followup", False),
            "has_prior": bool(ts.get("prior_study")),
            "num_steps": agent_pred.get("num_steps", 0),
            "wall_time_ms": agent_pred.get("wall_time_ms", 0),
            "has_ritl_rerun": sid in _predictions.get("agent_initial_ritl_rerun", {}),
            "has_ritl_checkpoint": sid in _predictions.get("agent_initial_ritl_checkpoint", {}),
        }
        studies.append(study)

    if dataset:
        studies = [s for s in studies if s["dataset"] == dataset]
    if ritl_only:
        studies = [s for s in studies if s["has_ritl_rerun"] or s["has_ritl_checkpoint"]]
    return {
        "studies": studies,
        "total": len(studies),
        "datasets": sorted(set(s["dataset"] for s in studies)),
        "models": _model_names,
    }


@router.get("/results/{study_id}")
async def get_result(study_id: str):
    """Full detail for one study: ground truth, all predictions, trajectory."""
    # Try eval_v4 test set first, then fall back to full datasets
    ts = _test_set_by_id.get(study_id) or _all_studies_by_id.get(study_id)
    if not ts:
        raise HTTPException(404, f"Study not found: {study_id}")

    # Collect predictions from all models
    predictions = {}
    for model_name, preds_by_id in _predictions.items():
        if study_id in preds_by_id:
            predictions[model_name] = preds_by_id[study_id]

    # Collect trajectories
    trajectories = {}
    for model_name, trajs_by_id in _trajectories.items():
        if study_id in trajs_by_id:
            trajectories[model_name] = trajs_by_id[study_id]

    return {
        "study": ts,
        "predictions": predictions,
        "trajectories": trajectories,
    }


@router.get("/scores")
async def get_scores(model: str | None = None, section: str | None = None):
    """Metric scores summary."""
    scores = _scores
    if model:
        scores = [s for s in scores if s["model"] == model]
    if section:
        scores = [s for s in scores if s["section"] == section]
    return {"scores": scores, "models": _model_names}
