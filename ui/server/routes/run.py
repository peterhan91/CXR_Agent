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

INITIAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config_combo_full.yaml"
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


DATA_DIR = PROJECT_ROOT / "data" / "eval"
_DATASET_FILES = [
    "mimic_cxr_test.json",
    "iu_xray_test.json",
    "chexpert_plus_valid.json",
    "rexgradient_test.json",
]


def _load_test_set():
    global _test_set_by_id
    if not _test_set_by_id:
        # Load eval_v4 test set first
        with open(TEST_SET_PATH) as f:
            for s in json.load(f):
                _test_set_by_id[s["study_id"]] = s
        # Then load all full datasets
        for fname in _DATASET_FILES:
            p = DATA_DIR / fname
            if p.exists():
                with open(p) as f:
                    for s in json.load(f):
                        if s["study_id"] not in _test_set_by_id:
                            _test_set_by_id[s["study_id"]] = s
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
    mode: str = "agent"  # "agent", "agent_guided", "chexagent2", "chexone", "medgemma"


class FeedbackRequest(BaseModel):
    feedback: str


# ── Background runners ────────────────────────────────────────────────────────

def _run_agent_background(run_id: str, study: dict, guided: bool = False):
    """Run the agent in a background thread. If guided=True, pause after 2 iterations for checkpoint."""
    run = _runs[run_id]
    try:
        run["status"] = "running"
        run["message"] = "Initializing agent..."

        agent = _get_agent()
        image_path = study["image_path"]

        # Prior study context (temporal comparison)
        prior_report = ""
        prior_image_path = ""
        if study.get("prior_study"):
            prior_report = study["prior_study"].get("report", "")
            prior_image_path = study["prior_study"].get("image_path", "")

        # Clinical metadata context
        # NOTE: Only reads metadata fields — NEVER reads report_gt, findings, or impression
        clinical_context = ""
        meta = study.get("metadata") or {}
        ctx_parts = []
        age = meta.get("age") or (
            (meta.get("admission_info") or {}).get("demographics", {}).get("age")
        )
        sex = meta.get("sex") or (
            (meta.get("admission_info") or {}).get("demographics", {}).get("gender")
        )
        if age is not None and sex:
            sex_word = "male" if str(sex).upper() in ("M", "MALE") else "female"
            ctx_parts.append(f"Patient: {age} year old {sex_word}")
        indication = (meta.get("indication") or "").strip()
        if indication:
            ctx_parts.append(f"Indication: {indication}")
        comparison = (meta.get("comparison") or "").strip()
        if comparison and comparison not in ("___", "___."):
            ctx_parts.append(f"Comparison: {comparison}")
        if ctx_parts:
            clinical_context = "\n".join(ctx_parts)

        # Lateral view
        lateral_image_path = study.get("lateral_image_path") or ""

        run["message"] = "Agent running (guided)..." if guided else "Agent running..."
        t0 = time.time()

        # Monkey-patch the agent to capture the trajectory object mid-run
        # so we can stream partial steps via the poll endpoint
        original_react_loop = agent._react_loop
        original_max_iters = agent.max_iterations

        if guided:
            # Guided mode: let the agent investigate fully, but intercept
            # right before it writes the final report (when it stops calling tools).
            # We do this by patching _react_loop to set force_report_on_max=False
            # AND by intercepting the "no tool calls" branch: when the agent
            # decides to write its report, we save the draft but mark as checkpoint.
            def _patched_react_loop(messages, system_prompt, traj, max_iters, **kwargs):
                run["_live_trajectory"] = traj
                # Run the loop but don't force report — let it stop naturally
                # We intercept the final report in the trajectory after the loop
                return original_react_loop(messages, system_prompt, traj, max_iters, force_report_on_max=False)
        else:
            def _patched_react_loop(messages, system_prompt, traj, max_iters, **kwargs):
                run["_live_trajectory"] = traj
                return original_react_loop(messages, system_prompt, traj, max_iters, **kwargs)

        agent._react_loop = _patched_react_loop

        trajectory = agent.run(
            image_path=image_path,
            concept_prior_text="",
            image_id=study["study_id"],
            prior_report=prior_report,
            prior_image_path=prior_image_path,
            clinical_context=clinical_context,
            lateral_image_path=lateral_image_path,
        )
        agent._react_loop = original_react_loop
        agent.max_iterations = original_max_iters

        wall_time = (time.time() - t0) * 1000

        # Format steps — only include actual tool calls
        steps = []
        for s in trajectory.steps:
            if isinstance(s, dict) and s.get("type") != "tool_call":
                continue
            steps.append({
                "iteration": s.get("iteration", 0) if isinstance(s, dict) else 0,
                "type": "tool_call",
                "tool_name": s.get("tool_name", "") if isinstance(s, dict) else s.tool_name,
                "tool_input": s.get("tool_input", {}) if isinstance(s, dict) else s.tool_input,
                "tool_output": s.get("tool_output", s.get("output", "")) if isinstance(s, dict) else s.output,
                "duration_ms": s.get("duration_ms", 0) if isinstance(s, dict) else s.duration_ms,
                "parallel_count": s.get("parallel_count", 1) if isinstance(s, dict) else 1,
            })

        if guided:
            # Generate a checkpoint summary for the attending
            draft = trajectory.final_report or ""
            if draft:
                try:
                    import anthropic as _anth
                    _client = _anth.Anthropic()
                    summary_resp = _client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=500,
                        temperature=0.0,
                        system="You are presenting CXR findings to an attending radiologist for review. Be concise and clinical.",
                        messages=[{"role": "user", "content": (
                            f"The AI agent investigated this chest X-ray using multiple tools and produced this draft report:\n\n"
                            f"{draft}\n\n"
                            f"Summarize the key findings as a bullet list for attending review. "
                            f"Flag any findings where tool agreement was low or uncertain. "
                            f"End with: 'Please review and provide feedback before I finalize.'"
                        )}],
                    )
                    checkpoint_summary = summary_resp.content[0].text
                except Exception as e:
                    logger.warning(f"Failed to generate checkpoint summary: {e}")
                    checkpoint_summary = draft
            else:
                checkpoint_summary = "Agent completed investigation but did not produce a draft. Review tool outputs and provide guidance."

            run["status"] = "awaiting_feedback"
            run["message"] = checkpoint_summary
        else:
            run["status"] = "complete"
            run["message"] = "Done"

        # Strip GROUNDINGS section from report for clean display
        raw_report = trajectory.final_report or ""
        try:
            from scripts.eval_mimic import strip_groundings
            clean_report, _ = strip_groundings(raw_report)
        except Exception:
            clean_report = raw_report

        run["result"] = {
            "study_id": study["study_id"],
            "report_pred": clean_report,
            "report_pred_raw": raw_report,
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
    """Continue an agent run with RITL feedback. Supports multiple rounds."""
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

        # Monkey-patch to capture live trajectory for streaming
        original_react_loop = agent._react_loop

        def _patched_react_loop(messages, system_prompt, traj, max_iters, **kwargs):
            run["_live_ritl_trajectory"] = traj
            return original_react_loop(messages, system_prompt, traj, max_iters, **kwargs)

        agent._react_loop = _patched_react_loop

        trajectory = agent.continue_with_feedback(
            messages=parent_messages,
            system_prompt=parent_system,
            feedback=feedback,
        )

        agent._react_loop = original_react_loop

        wall_time = (time.time() - t0) * 1000

        steps = []
        for s in trajectory.steps:
            if isinstance(s, dict) and s.get("type") != "tool_call":
                continue
            steps.append({
                "iteration": s.get("iteration", 0) if isinstance(s, dict) else 0,
                "type": "tool_call",
                "tool_name": s.get("tool_name", "") if isinstance(s, dict) else s.tool_name,
                "tool_input": s.get("tool_input", {}) if isinstance(s, dict) else s.tool_input,
                "tool_output": s.get("tool_output", s.get("output", "")) if isinstance(s, dict) else s.output,
                "duration_ms": s.get("duration_ms", 0) if isinstance(s, dict) else s.duration_ms,
                "parallel_count": s.get("parallel_count", 1) if isinstance(s, dict) else 1,
            })

        # Track feedback rounds
        round_num = run.get("_feedback_round", 0) + 1
        run["_feedback_round"] = round_num

        # Append to existing RITL steps (for multiple rounds)
        prev_ritl_steps = run.get("ritl_trajectory_steps") or []
        all_ritl_steps = prev_ritl_steps + steps

        # Update the original result with the revised report
        prev_report = run.get("ritl_result", {}).get("report_pred") or (run.get("result", {}) or {}).get("report_pred", "")

        run["ritl_result"] = {
            "study_id": run["study_id"],
            "report_pred": trajectory.final_report,
            "report_pred_prev": prev_report,
            "feedback": feedback,
            "round": round_num,
            "num_steps": len(steps),
            "wall_time_ms": wall_time,
        }
        run["ritl_trajectory_steps"] = all_ritl_steps
        run.pop("_live_ritl_trajectory", None)
        run["status"] = "complete"
        run["message"] = f"RITL revision complete (round {round_num})"

        # Update messages for next round
        run["_messages"] = trajectory.messages
        run["_system_prompt"] = trajectory.system_prompt

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

    if req.mode not in ("agent", "agent_guided", "chexagent2", "chexone", "medgemma"):
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

    if req.mode in ("agent", "agent_guided"):
        guided = req.mode == "agent_guided"
        thread = threading.Thread(
            target=_run_agent_background, args=(run_id, study, guided), daemon=True
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

    # Build live partial steps if agent is still running
    trajectory = run.get("trajectory")
    if not trajectory and run.get("_live_trajectory"):
        live_traj = run["_live_trajectory"]
        live_steps = []
        for s in live_traj.steps:
            if not isinstance(s, dict) or s.get("type") != "tool_call":
                continue
            live_steps.append({
                "iteration": s.get("iteration", 0),
                "type": "tool_call",
                "tool_name": s.get("tool_name", ""),
                "tool_input": s.get("tool_input", {}),
                "tool_output": s.get("tool_output", ""),
                "duration_ms": s.get("duration_ms", 0),
                "parallel_count": s.get("parallel_count", 1),
            })
        if live_steps:
            trajectory = {
                "study_id": run["study_id"],
                "steps": live_steps,
                "num_steps": len(live_steps),
                "wall_time_ms": (time.time() - run["started_at"]) * 1000,
            }
            run["message"] = f"Running... ({len(live_steps)} tool calls)"

    # Build live RITL steps if feedback re-reasoning is in progress
    ritl_steps = run.get("ritl_trajectory_steps")
    if not ritl_steps and run.get("_live_ritl_trajectory"):
        live_ritl = run["_live_ritl_trajectory"]
        live_ritl_steps = []
        for s in live_ritl.steps:
            if not isinstance(s, dict) or s.get("type") != "tool_call":
                continue
            live_ritl_steps.append({
                "iteration": s.get("iteration", 0),
                "type": "tool_call",
                "tool_name": s.get("tool_name", ""),
                "tool_input": s.get("tool_input", {}),
                "tool_output": s.get("tool_output", ""),
                "duration_ms": s.get("duration_ms", 0),
                "parallel_count": s.get("parallel_count", 1),
            })
        if live_ritl_steps:
            ritl_steps = live_ritl_steps
            run["message"] = f"Re-reasoning... ({len(live_ritl_steps)} tool calls)"

    return {
        "run_id": run["run_id"],
        "study_id": run["study_id"],
        "mode": run["mode"],
        "status": run["status"],
        "message": run["message"],
        "result": run.get("result"),
        "trajectory": trajectory,
        "ritl_result": run.get("ritl_result"),
        "ritl_trajectory_steps": ritl_steps,
        "elapsed_ms": (time.time() - run["started_at"]) * 1000,
    }


@router.post("/run/{run_id}/feedback")
async def submit_feedback(run_id: str, req: FeedbackRequest):
    """Submit RITL feedback to continue an agent run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Run not found: {run_id}")
    if run["mode"] not in ("agent", "agent_guided"):
        raise HTTPException(400, "Feedback only supported for agent runs")
    if run["status"] not in ("complete", "awaiting_feedback"):
        raise HTTPException(400, "Agent run must be complete or at checkpoint before feedback")

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
        "whisper": "http://localhost:8011/health",
    }
    status = {}
    for name, url in servers.items():
        try:
            resp = requests.get(url, timeout=3)
            status[name] = resp.status_code == 200
        except Exception:
            status[name] = False
    return {"servers": status, "all_healthy": all(status.values())}
