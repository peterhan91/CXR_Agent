"""
/api/run — execute agent or baseline runs on a study.

Runs happen in background threads. Client polls /api/run/{run_id} for status.
"""

import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── In-memory run store ───────────────────────────────────────────────────────

_runs: dict[str, dict] = {}

# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config_initial.yaml"
TEST_SET_PATH = PROJECT_ROOT / "results" / "eval_v4" / "test_set.json"

# Baseline server endpoints
BASELINE_ENDPOINTS = {
    "chexagent2": {"url": "http://localhost:8001/generate_report", "payload_key": "image_path"},
    "chexone": {"url": "http://localhost:8002/generate_report", "payload_key": "image_path"},
    "medgemma": {"url": "http://localhost:8010/generate_report", "payload_key": "image_path"},
}

# Lazy-loaded agent components
_agent = None
_test_set_by_id: dict[str, dict] = {}


def _load_test_set():
    global _test_set_by_id
    if not _test_set_by_id:
        with open(TEST_SET_PATH) as f:
            data = json.load(f)
        _test_set_by_id = {s["study_id"]: s for s in data}
    return _test_set_by_id


def _get_agent():
    """Lazy-init the agent with config_initial.yaml settings."""
    global _agent
    if _agent is not None:
        return _agent

    import yaml
    from scripts.run_agent import build_tools
    from agent.react_agent import CXRReActAgent

    with open(INITIAL_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    agent_cfg = config.get("agent", {})
    prompt_mode = agent_cfg.get("prompt_mode", "initial")
    tools = build_tools(config, prompt_mode=prompt_mode)

    _agent = CXRReActAgent(
        model=agent_cfg.get("model", "claude-sonnet-4-6"),
        max_iterations=agent_cfg.get("max_iterations", 10),
        max_tokens=agent_cfg.get("max_tokens", 4096),
        temperature=agent_cfg.get("temperature", 0.0),
        tools=tools,
        use_skills=config.get("skill", {}).get("enabled", False),
        prompt_mode=prompt_mode,
    )
    logger.info(f"Agent initialized: {len(tools)} tools, mode={prompt_mode}")
    return _agent


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    study_id: str
    mode: str = "agent"  # "agent", "chexagent2", "chexone", "medgemma"


class FeedbackRequest(BaseModel):
    feedback: str


# ── Background runners ────────────────────────────────────────────────────────

def _run_agent_background(run_id: str, study: dict):
    """Run the full agent in a background thread."""
    run = _runs[run_id]
    try:
        run["status"] = "running"
        run["message"] = "Initializing agent..."

        agent = _get_agent()
        image_path = study["image_path"]
        prior_report = ""
        prior_image_path = ""
        if study.get("prior_study"):
            prior_report = study["prior_study"].get("report", "")
            prior_image_path = study["prior_study"].get("image_path", "")

        run["message"] = "Agent running..."
        t0 = time.time()

        trajectory = agent.run(
            image_path=image_path,
            concept_prior_text="",
            image_id=study["study_id"],
            prior_report=prior_report,
            prior_image_path=prior_image_path,
        )

        wall_time = (time.time() - t0) * 1000

        # Format steps
        steps = []
        for s in trajectory.steps:
            steps.append({
                "iteration": s.get("iteration", 0) if isinstance(s, dict) else 0,
                "type": "tool_call",
                "tool_name": s.get("tool_name", "") if isinstance(s, dict) else s.tool_name,
                "tool_input": s.get("tool_input", {}) if isinstance(s, dict) else s.tool_input,
                "tool_output": s.get("output", "") if isinstance(s, dict) else s.output,
                "duration_ms": s.get("duration_ms", 0) if isinstance(s, dict) else s.duration_ms,
            })

        run["status"] = "complete"
        run["message"] = "Done"
        run["result"] = {
            "study_id": study["study_id"],
            "report_pred": trajectory.final_report,
            "report_pred_raw": trajectory.final_report,
            "num_steps": len(trajectory.steps),
            "wall_time_ms": wall_time,
            "input_tokens": trajectory.total_input_tokens,
            "output_tokens": trajectory.total_output_tokens,
            "unused_tools": trajectory.unused_tools,
        }
        run["trajectory"] = {
            "study_id": study["study_id"],
            "steps": steps,
            "num_steps": len(steps),
            "wall_time_ms": wall_time,
            "input_tokens": trajectory.total_input_tokens,
            "output_tokens": trajectory.total_output_tokens,
        }
        # Store messages for RITL continuation
        run["_messages"] = trajectory.messages
        run["_system_prompt"] = trajectory.system_prompt

    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        run["status"] = "error"
        run["message"] = str(e)


def _run_baseline_background(run_id: str, study: dict, model: str):
    """Run a baseline model (single server call) in a background thread."""
    run = _runs[run_id]
    try:
        run["status"] = "running"
        run["message"] = f"Calling {model} server..."

        endpoint = BASELINE_ENDPOINTS[model]
        t0 = time.time()

        payload = {"image_path": study["image_path"]}
        if model == "chexone":
            payload["reasoning"] = False

        resp = requests.post(endpoint["url"], json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        wall_time = (time.time() - t0) * 1000
        report = data.get("report", "")

        run["status"] = "complete"
        run["message"] = "Done"
        run["result"] = {
            "study_id": study["study_id"],
            "report_pred": report,
            "wall_time_ms": wall_time,
        }

    except requests.ConnectionError:
        run["status"] = "error"
        run["message"] = f"{model} server not reachable. Is it running?"
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        run["status"] = "error"
        run["message"] = str(e)


def _run_ritl_feedback_background(run_id: str, feedback: str):
    """Continue an agent run with RITL feedback."""
    run = _runs[run_id]
    parent_messages = run.get("_messages", [])
    parent_system = run.get("_system_prompt", "")

    if not parent_messages:
        run["status"] = "error"
        run["message"] = "No agent messages to continue from"
        return

    try:
        run["status"] = "running"
        run["message"] = "Agent re-reasoning with feedback..."

        agent = _get_agent()
        t0 = time.time()

        # Inject feedback as user message and continue the loop
        import copy
        messages = copy.deepcopy(parent_messages)
        messages.append({"role": "user", "content": feedback})

        # Re-enter the agent loop manually
        from agent.react_agent import AgentTrajectory
        trajectory = AgentTrajectory(
            image_id=run["study_id"],
            concept_prior="",
            messages=messages,
            system_prompt=parent_system,
        )

        # Use the agent's internal loop
        trajectory = agent._continue_loop(
            messages=messages,
            system_prompt=parent_system,
            trajectory=trajectory,
        )

        wall_time = (time.time() - t0) * 1000

        steps = []
        for s in trajectory.steps:
            steps.append({
                "iteration": s.get("iteration", 0) if isinstance(s, dict) else 0,
                "type": "tool_call",
                "tool_name": s.get("tool_name", "") if isinstance(s, dict) else s.tool_name,
                "tool_input": s.get("tool_input", {}) if isinstance(s, dict) else s.tool_input,
                "tool_output": s.get("output", "") if isinstance(s, dict) else s.output,
                "duration_ms": s.get("duration_ms", 0) if isinstance(s, dict) else s.duration_ms,
            })

        run["ritl_status"] = "complete"
        run["ritl_result"] = {
            "study_id": run["study_id"],
            "report_pred": trajectory.final_report,
            "feedback": feedback,
            "num_steps": len(trajectory.steps),
            "wall_time_ms": wall_time,
        }
        run["ritl_trajectory_steps"] = steps
        run["status"] = "complete"
        run["message"] = "RITL revision complete"

    except Exception as e:
        logger.error(f"RITL {run_id} failed: {e}", exc_info=True)
        run["status"] = "error"
        run["message"] = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/run")
async def start_run(req: RunRequest):
    """Start an agent or baseline run on a study."""
    test_set = _load_test_set()
    study = test_set.get(req.study_id)
    if not study:
        raise HTTPException(404, f"Study not found: {req.study_id}")

    if req.mode not in ("agent", "chexagent2", "chexone", "medgemma"):
        raise HTTPException(400, f"Invalid mode: {req.mode}")

    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "run_id": run_id,
        "study_id": req.study_id,
        "mode": req.mode,
        "status": "queued",
        "message": "Starting...",
        "result": None,
        "trajectory": None,
        "started_at": time.time(),
    }

    if req.mode == "agent":
        thread = threading.Thread(
            target=_run_agent_background, args=(run_id, study), daemon=True
        )
    else:
        thread = threading.Thread(
            target=_run_baseline_background, args=(run_id, study, req.mode), daemon=True
        )
    thread.start()

    return {"run_id": run_id, "status": "queued"}


@router.get("/run/{run_id}")
async def get_run(run_id: str):
    """Poll run status and results."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Run not found: {run_id}")

    # Don't expose internal state
    return {
        "run_id": run["run_id"],
        "study_id": run["study_id"],
        "mode": run["mode"],
        "status": run["status"],
        "message": run["message"],
        "result": run.get("result"),
        "trajectory": run.get("trajectory"),
        "ritl_result": run.get("ritl_result"),
        "ritl_trajectory_steps": run.get("ritl_trajectory_steps"),
        "elapsed_ms": (time.time() - run["started_at"]) * 1000,
    }


@router.post("/run/{run_id}/feedback")
async def submit_feedback(run_id: str, req: FeedbackRequest):
    """Submit RITL feedback to continue an agent run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Run not found: {run_id}")
    if run["mode"] != "agent":
        raise HTTPException(400, "Feedback only supported for agent runs")
    if run["status"] != "complete":
        raise HTTPException(400, "Agent run must be complete before feedback")

    run["status"] = "running"
    run["message"] = "Processing feedback..."

    thread = threading.Thread(
        target=_run_ritl_feedback_background, args=(run_id, req.feedback), daemon=True
    )
    thread.start()

    return {"status": "running", "message": "Feedback submitted"}


@router.get("/servers/health")
async def check_servers():
    """Check which tool servers are reachable."""
    servers = {
        "chexagent2": "http://localhost:8001/health",
        "chexone": "http://localhost:8002/health",
        "biomedparse": "http://localhost:8005/health",
        "factchexcker": "http://localhost:8007/health",
        "cxr_foundation": "http://localhost:8008/health",
        "chexzero": "http://localhost:8009/health",
        "medgemma": "http://localhost:8010/health",
    }
    status = {}
    for name, url in servers.items():
        try:
            resp = requests.get(url, timeout=3)
            status[name] = resp.status_code == 200
        except Exception:
            status[name] = False
    return {"servers": status, "all_healthy": all(status.values())}
