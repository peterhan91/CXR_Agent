#!/usr/bin/env python3
"""
Radiologist-in-the-Loop (RITL) Simulation.

Uses ground-truth reports as oracle radiologist feedback to measure whether
injecting corrections into the agent pipeline closes the gap to GT.

Two modes:

  # Exp 1: Text-only revision (no servers needed, ~5 min)
  python scripts/eval_ritl.py --mode text --output results/eval_v4/

  # Exp 2: Agent re-run with feedback (needs servers, uses tool cache)
  python scripts/eval_ritl.py --mode rerun --output results/eval_v4/

Output: predictions_{base}_ritl_text.json / predictions_{base}_ritl_rerun.json
Score with the existing pipeline:
  python scripts/eval_mimic.py --mode score --output results/eval_v4/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from scripts.eval_mimic import (
    _load_test_set,
    _load_existing_predictions,
    _save_predictions,
    strip_groundings,
)

logger = logging.getLogger(__name__)

# ─── Evidence Extraction ─────────────────────────────────────────────────────


def extract_tool_evidence(traj_record: dict) -> str:
    """Extract tool outputs from a trajectory record as structured evidence.

    Reads trajectory steps, concatenates tool outputs in a readable format:
      [Report: CheXagent-2] <output>
      [VQA: CheXagent-2] Q: ... A: ...
    """
    evidence_parts = []
    for step in traj_record.get("steps", []):
        if step.get("type") != "tool_call":
            continue
        tool_name = step["tool_name"]
        output = step["tool_output"]
        # Categorize by tool type for readability
        if "report" in tool_name:
            label = f"Report: {tool_name}"
        elif "vqa" in tool_name:
            label = f"VQA: {tool_name}"
        elif "classify" in tool_name:
            label = f"Classify: {tool_name}"
        elif "grounding" in tool_name:
            label = f"Grounding: {tool_name}"
        elif "segment" in tool_name:
            label = f"Segment: {tool_name}"
        elif "verify" in tool_name:
            label = f"Verify: {tool_name}"
        else:
            label = tool_name
        evidence_parts.append(f"[{label}]\n{output}")
    return "\n\n".join(evidence_parts)


# ─── Structured Critique Generation ──────────────────────────────────────────

CRITIQUE_SYSTEM_PROMPT = """\
You are an attending radiologist. Give feedback in 1-2 SHORT sentences like you'd say walking past the resident's workstation. 15-25 words total.

Good examples:
"I don't buy the pneumothorax, re-examine the right apex."
"The edema's worse than mild. Also take another look at the right upper lobe."
"Heart size looks normal to me, and look at the apex again."

Bad examples (TOO LONG):
"The pneumothorax is the one I'd push back on — take another look at the right apex and make sure you're not overcalling that, especially post-procedure with overlying soft tissue."

Rules:
- 1-2 sentences, 15-25 words MAX. No hedging, no explanations, no politeness.
- You may dispute severity or question a finding the resident already mentioned.
- For missed findings: point to a REGION ("look at the right upper lobe again").
  Do NOT name the finding or say "you missed something."
- Never say "reference", "ground truth", or "the reference suggests."
- Speak as if YOU looked at the film, not as if you read a report.
- Plain text only. No markdown, no bullets."""


def generate_structured_critique(
    draft_report: str,
    gt_report: str,
    client: "anthropic.Anthropic",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Use Claude to generate attending-style directional feedback.

    Compares draft against GT and produces clinical hints — never reveals
    the GT verbatim. The agent must re-reason with tools to act on feedback.
    """
    user_message = (
        f"RESIDENT'S REPORT:\n{draft_report}\n\n"
        f"REFERENCE (do NOT reveal):\n{gt_report}\n\n"
        "Give 1-2 short sentences of feedback."
    )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call(**kwargs):
        return client.messages.create(**kwargs)

    response = _call(
        model=model,
        max_tokens=150,
        temperature=0.0,
        system=CRITIQUE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text_blocks = [b for b in response.content if b.type == "text"]
    critique = "\n".join(b.text for b in text_blocks)
    return critique, response.usage.input_tokens, response.usage.output_tokens


def format_gt_feedback(
    gt_report: str,
    feedback_mode: str = "structured",
    draft_report: str = "",
    client: "anthropic.Anthropic" = None,
    model: str = "claude-sonnet-4-6",
) -> tuple:
    """Format ground-truth as radiologist feedback.

    Args:
        gt_report: Full GT report text
        feedback_mode: 'structured' (default) generates directional critique;
                       'full_gt' gives verbatim GT (for training data / ceiling)
        draft_report: Agent's draft (needed for structured mode)
        client: Anthropic client (needed for structured mode)
        model: Model for critique generation

    Returns:
        (feedback_text, extra_input_tokens, extra_output_tokens)
    """
    if feedback_mode == "structured":
        if not client or not draft_report:
            raise ValueError("structured mode requires client and draft_report")
        critique, in_tok, out_tok = generate_structured_critique(
            draft_report, gt_report, client, model,
        )
        feedback = (
            "Attending feedback: " + critique + "\n"
            "Re-examine with your tools and revise your report."
        )
        return feedback, in_tok, out_tok

    if feedback_mode == "full_gt":
        feedback = (
            "The attending radiologist has reviewed your report and provided "
            "the following corrected report. Please revise your report to "
            "incorporate these corrections while preserving any tool-verified "
            "details from your original analysis.\n\n"
            f"CORRECTED REPORT:\n{gt_report}"
        )
        return feedback, 0, 0

    raise ValueError(f"Unknown feedback_mode: {feedback_mode}")


# ─── Revision System Prompt ──────────────────────────────────────────────────

REVISION_SYSTEM_PROMPT = """\
You are an expert radiologist revising a chest X-ray report based on attending feedback.

You will receive:
1. TOOL EVIDENCE: outputs from CXR analysis tools (reports, VQA, classifications)
2. DRAFT REPORT: the original AI-generated report
3. ATTENDING FEEDBACK: corrections or guidance from the attending radiologist

Your task: revise the draft to address the attending's feedback. Ground your \
revisions in the tool evidence where possible. Do not fabricate findings.

Output ONLY the revised report in this exact format (no preamble, no markdown, no explanation):

FINDINGS:
<findings text>

IMPRESSION:
<impression text>"""


# ─── Experiment 1: Text-only Revision ────────────────────────────────────────


def run_text_revision(args):
    """Exp 1: Single Claude call per study to revise draft with GT feedback.

    No servers needed. Uses prior tool evidence from trajectories + GT feedback.
    """
    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir, args)
    gt_by_study = {e["study_id"]: e for e in test_set}

    # Load baseline predictions (the drafts to revise)
    base_name = args.predictions_base
    base_path = output_dir / f"predictions_{base_name}.json"
    if not base_path.exists():
        logger.error(f"Baseline predictions not found: {base_path}")
        sys.exit(1)
    with open(base_path) as f:
        base_preds = {p["study_id"]: p for p in json.load(f)}
    logger.info(f"Loaded {len(base_preds)} baseline predictions from {base_path}")

    # Load trajectories for tool evidence
    traj_path = output_dir / f"trajectories_{base_name}.jsonl"
    traj_by_study = {}
    if traj_path.exists():
        with open(traj_path) as f:
            for line in f:
                rec = json.loads(line)
                traj_by_study[rec["study_id"]] = rec
        logger.info(f"Loaded {len(traj_by_study)} trajectories from {traj_path}")
    else:
        logger.warning(f"No trajectories found at {traj_path} — revising without tool evidence")

    # Output
    predictions_path = output_dir / f"predictions_{base_name}_ritl_text.json"
    existing = _load_existing_predictions(predictions_path)
    predictions = list(existing.values())

    client = anthropic.Anthropic()
    model = args.model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call(**kwargs):
        return client.messages.create(**kwargs)

    total = len(test_set)
    cum_in = sum(p.get("input_tokens", 0) for p in predictions)
    cum_out = sum(p.get("output_tokens", 0) for p in predictions)
    errors = 0
    t_start = time.time()

    if args.max_samples:
        test_set = test_set[:args.max_samples]
        total = len(test_set)

    for i, entry in enumerate(test_set):
        study_id = entry["study_id"]
        if study_id in existing:
            continue

        # Get draft report
        base_pred = base_preds.get(study_id)
        if not base_pred:
            logger.warning(f"[{i+1}/{total}] No baseline prediction for {study_id}, skipping")
            continue
        draft_report = base_pred.get("report_pred", "")
        if not draft_report.strip():
            logger.warning(f"[{i+1}/{total}] Empty draft for {study_id}, skipping")
            continue

        # Get GT report for feedback
        gt_report = entry.get("report_gt", "")
        if not gt_report.strip():
            continue

        # Get tool evidence from trajectory
        traj = traj_by_study.get(study_id, {})
        evidence = extract_tool_evidence(traj)

        # Generate feedback (structured critique or full GT)
        feedback, fb_in, fb_out = format_gt_feedback(
            gt_report, args.feedback_mode,
            draft_report=draft_report, client=client, model=model,
        )

        # Build revision prompt
        user_parts = []
        if evidence:
            user_parts.append(f"TOOL EVIDENCE:\n{evidence}")
        user_parts.append(f"DRAFT REPORT:\n{draft_report}")
        user_parts.append(f"ATTENDING FEEDBACK:\n{feedback}")
        user_message = "\n\n---\n\n".join(user_parts)

        logger.info(f"[{i+1}/{total}] Revising study {study_id}")
        t0 = time.time()

        try:
            response = _call(
                model=model,
                max_tokens=2048,
                temperature=0.0,
                system=REVISION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text_blocks = [b for b in response.content if b.type == "text"]
            revised = "\n".join(b.text for b in text_blocks)
            in_tok = response.usage.input_tokens + fb_in
            out_tok = response.usage.output_tokens + fb_out
        except Exception as e:
            logger.error(f"Revision failed for {study_id}: {e}", exc_info=True)
            revised = ""
            in_tok = fb_in
            out_tok = fb_out
            errors += 1

        wall_ms = (time.time() - t0) * 1000
        cum_in += in_tok
        cum_out += out_tok

        # Strip any preamble/groundings from revised report
        clean_report, groundings = strip_groundings(revised)

        pred = {
            "study_id": study_id,
            "report_pred": clean_report,
            "report_pred_raw": revised,
            "groundings": groundings,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "wall_time_ms": wall_ms,
            "base_model": base_name,
            "feedback": feedback,
        }
        predictions.append(pred)
        existing[study_id] = pred

        # Save every 10 studies
        if len(predictions) % 10 == 0:
            _save_predictions(predictions_path, predictions)
            done = len(predictions)
            elapsed = time.time() - t_start
            logger.info(
                f"[{done}/{total}] saved | "
                f"tokens: {cum_in:,} in / {cum_out:,} out | "
                f"{elapsed/60:.1f}min elapsed"
            )

    _save_predictions(predictions_path, predictions)
    print(f"\nRITL text revision: {len(predictions)} predictions -> {predictions_path}")
    print(f"Total tokens: {cum_in:,} input, {cum_out:,} output")
    if errors:
        print(f"Errors: {errors}")


# ─── Experiment 2: Agent Re-run with Feedback ────────────────────────────────


def run_agent_rerun(args):
    """Exp 2: Run agent fresh, then inject GT feedback and continue.

    Needs servers running. Tool cache makes repeated image analysis fast.
    """
    import yaml
    from agent.react_agent import CXRReActAgent

    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir, args)
    gt_by_study = {e["study_id"]: e for e in test_set}

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    acfg = config.get("agent", {})

    # Load baseline predictions to check which studies had agent runs
    base_name = args.predictions_base
    base_path = output_dir / f"predictions_{base_name}.json"
    base_preds = {}
    if base_path.exists():
        with open(base_path) as f:
            base_preds = {p["study_id"]: p for p in json.load(f)}

    # CLEAR scorer
    clear_cfg = config.get("clear", {})
    scorer = None
    if clear_cfg.get("enabled", True):
        from clear.concept_scorer import CLEARConceptScorer
        scorer = CLEARConceptScorer(
            model_path=clear_cfg.get("model_path"),
            concepts_path=clear_cfg.get("concepts_path"),
            dinov2_model_name=clear_cfg.get("dinov2_model_name", "dinov2_vitb14"),
            image_resolution=clear_cfg.get("image_resolution", 448),
        )
        logger.info("Loading CLEAR model...")
        scorer.load()
        logger.info("CLEAR model ready")

    # Build tools
    from scripts.eval_mimic import _build_tools
    prompt_mode = acfg.get("prompt_mode", "current")
    tools = _build_tools(config, prompt_mode=prompt_mode)

    use_skills = config.get("skill", {}).get("enabled", True)
    agent = CXRReActAgent(
        model=acfg.get("model", "claude-sonnet-4-6"),
        max_iterations=acfg.get("max_iterations", 10),
        max_tokens=acfg.get("max_tokens", 4096),
        temperature=acfg.get("temperature", 0.0),
        tools=tools,
        reasoning_effort=acfg.get("reasoning_effort"),
        use_skills=use_skills,
        prompt_mode=prompt_mode,
    )

    # Output
    predictions_path = output_dir / f"predictions_{base_name}_ritl_rerun.json"
    trajectories_path = output_dir / f"trajectories_{base_name}_ritl_rerun.jsonl"
    existing = _load_existing_predictions(predictions_path)
    predictions = list(existing.values())

    total = len(test_set)
    cum_in = sum(p.get("input_tokens", 0) for p in predictions)
    cum_out = sum(p.get("output_tokens", 0) for p in predictions)
    errors = 0
    t_start = time.time()

    if args.max_samples:
        test_set = test_set[:args.max_samples]
        total = len(test_set)

    for i, entry in enumerate(test_set):
        study_id = entry["study_id"]
        if study_id in existing:
            continue

        gt_report = entry.get("report_gt", "")
        if not gt_report.strip():
            continue

        logger.info(f"[{i+1}/{total}] RITL rerun: study {study_id}")
        t0 = time.time()

        try:
            # Step 1: Run agent fresh
            concept_prior = ""
            if scorer:
                top_k = config.get("clear", {}).get("top_k", 20)
                prior_template = None
                if prompt_mode == "initial":
                    from agent.initial_mode import CONCEPT_PRIOR_TEMPLATE_INITIAL
                    prior_template = CONCEPT_PRIOR_TEMPLATE_INITIAL
                concept_prior = scorer.score_image(
                    entry["image_path"], top_k=top_k, template=prior_template,
                )

            # Get prior context if available
            prior_report = ""
            prior_image_path = ""
            prior_study = entry.get("prior_study")
            if prior_study and isinstance(prior_study, dict):
                prior_report = prior_study.get("report", "")
                prior_image_path = prior_study.get("image_path", "")

            trajectory = agent.run(
                image_path=entry["image_path"],
                concept_prior_text=concept_prior,
                image_id=study_id,
                prior_report=prior_report,
                prior_image_path=prior_image_path,
            )

            # Step 2: Generate feedback and inject
            draft_report = trajectory.final_report
            feedback, fb_in, fb_out = format_gt_feedback(
                gt_report, args.feedback_mode,
                draft_report=draft_report, client=agent.client, model=args.model,
            )
            trajectory.total_input_tokens += fb_in
            trajectory.total_output_tokens += fb_out

            trajectory = agent.continue_with_feedback(
                messages=trajectory.messages,
                system_prompt=trajectory.system_prompt,
                feedback=feedback,
                trajectory=trajectory,
                max_feedback_iterations=args.max_feedback_iterations,
            )

            report = trajectory.final_report
            in_tok = trajectory.total_input_tokens
            out_tok = trajectory.total_output_tokens
            steps = trajectory.steps

        except Exception as e:
            logger.error(f"RITL rerun failed for {study_id}: {e}", exc_info=True)
            report = ""
            in_tok = out_tok = 0
            steps = []
            errors += 1

        wall_ms = (time.time() - t0) * 1000
        cum_in += in_tok
        cum_out += out_tok

        clean_report, groundings = strip_groundings(report)

        pred = {
            "study_id": study_id,
            "report_pred": clean_report,
            "report_pred_raw": report,
            "groundings": groundings,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "num_steps": len(steps),
            "wall_time_ms": wall_ms,
            "base_model": base_name,
            "feedback": feedback,
        }
        predictions.append(pred)
        existing[study_id] = pred

        # Save trajectory
        traj_record = {
            "study_id": study_id,
            "concept_prior": concept_prior if scorer else "",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "num_steps": len(steps),
            "wall_time_ms": wall_ms,
            "groundings": groundings,
            "feedback": feedback,
            "steps": steps,
        }
        with open(trajectories_path, "a") as f:
            f.write(json.dumps(traj_record) + "\n")

        # Save every 5 studies
        if len(predictions) % 5 == 0:
            _save_predictions(predictions_path, predictions)
            done = len(predictions)
            elapsed = time.time() - t_start
            logger.info(
                f"[{done}/{total}] saved | "
                f"tokens: {cum_in:,} in / {cum_out:,} out | "
                f"{elapsed/60:.1f}min elapsed"
            )

    _save_predictions(predictions_path, predictions)
    print(f"\nRITL agent rerun: {len(predictions)} predictions -> {predictions_path}")
    print(f"Trajectories: {trajectories_path}")
    print(f"Total tokens: {cum_in:,} input, {cum_out:,} output")
    if errors:
        print(f"Errors: {errors}")


# ─── Checkpoint Critique ──────────────────────────────────────────────────────

CHECKPOINT_CRITIQUE_PROMPT = """\
You are an attending radiologist. A resident ran initial tools and is about to write the report. \
Give 1-2 SHORT sentences redirecting their investigation.

Good examples:
"I doubt the pneumothorax your tools flagged. Also look at the right apex more carefully."
"Heart looks normal to me despite what the tools say. The edema's worse than mild."
"Look at the right side again — there's more going on than bilateral findings."

Rules:
- 15-25 words MAX. Terse like a busy attending.
- You may dispute severity or question a finding the tools already flagged.
- For missed findings: point to a REGION only ("check the right apex", "look below the diaphragm").
  Do NOT name the finding itself ("there's a mass", "I see bowel loops").
- Never say "reference" or "ground truth."
- Plain text only. No markdown, no bullets."""


def generate_checkpoint_critique(
    tool_evidence: str,
    gt_report: str,
    client: "anthropic.Anthropic",
    model: str = "claude-sonnet-4-6",
) -> tuple:
    """Generate attending feedback based on tool outputs vs GT, before the report is written."""
    user_message = (
        f"TOOL OUTPUTS SO FAR:\n{tool_evidence}\n\n"
        f"REFERENCE (do NOT reveal):\n{gt_report}\n\n"
        "Give 1-2 short sentences of guidance before the resident writes the report."
    )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call(**kwargs):
        return client.messages.create(**kwargs)

    response = _call(
        model=model,
        max_tokens=150,
        temperature=0.0,
        system=CHECKPOINT_CRITIQUE_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text_blocks = [b for b in response.content if b.type == "text"]
    critique = "\n".join(b.text for b in text_blocks)
    return critique, response.usage.input_tokens, response.usage.output_tokens


# ─── Experiment 3: Checkpoint (mid-loop feedback) ────────────────────────────


def run_checkpoint(args):
    """Exp 3: Pause agent after initial tool calls, inject feedback, then continue.

    More clinically realistic — feedback arrives before the agent commits to a
    report, so it can redirect its verification strategy.
    """
    import yaml
    from agent.react_agent import CXRReActAgent

    output_dir = Path(args.output)
    test_set = _load_test_set(output_dir, args)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    acfg = config.get("agent", {})
    base_name = args.predictions_base
    checkpoint_after = args.checkpoint_after

    # CLEAR scorer
    clear_cfg = config.get("clear", {})
    scorer = None
    if clear_cfg.get("enabled", True):
        from clear.concept_scorer import CLEARConceptScorer
        scorer = CLEARConceptScorer(
            model_path=clear_cfg.get("model_path"),
            concepts_path=clear_cfg.get("concepts_path"),
            dinov2_model_name=clear_cfg.get("dinov2_model_name", "dinov2_vitb14"),
            image_resolution=clear_cfg.get("image_resolution", 448),
        )
        scorer.load()

    from scripts.eval_mimic import _build_tools
    prompt_mode = acfg.get("prompt_mode", "current")
    tools = _build_tools(config, prompt_mode=prompt_mode)

    use_skills = config.get("skill", {}).get("enabled", True)
    agent = CXRReActAgent(
        model=acfg.get("model", "claude-sonnet-4-6"),
        max_iterations=acfg.get("max_iterations", 10),
        max_tokens=acfg.get("max_tokens", 4096),
        temperature=acfg.get("temperature", 0.0),
        tools=tools,
        reasoning_effort=acfg.get("reasoning_effort"),
        use_skills=use_skills,
        prompt_mode=prompt_mode,
    )

    predictions_path = output_dir / f"predictions_{base_name}_ritl_checkpoint.json"
    trajectories_path = output_dir / f"trajectories_{base_name}_ritl_checkpoint.jsonl"
    existing = _load_existing_predictions(predictions_path)
    predictions = list(existing.values())

    total = len(test_set)
    cum_in = sum(p.get("input_tokens", 0) for p in predictions)
    cum_out = sum(p.get("output_tokens", 0) for p in predictions)
    errors = 0
    t_start = time.time()

    if args.max_samples:
        test_set = test_set[:args.max_samples]
        total = len(test_set)

    remaining_iterations = agent.max_iterations - checkpoint_after

    for i, entry in enumerate(test_set):
        study_id = entry["study_id"]
        if study_id in existing:
            continue

        gt_report = entry.get("report_gt", "")
        if not gt_report.strip():
            continue

        logger.info(f"[{i+1}/{total}] Checkpoint: study {study_id}")
        t0 = time.time()

        try:
            # CLEAR
            concept_prior = ""
            if scorer:
                top_k = config.get("clear", {}).get("top_k", 20)
                prior_template = None
                if prompt_mode == "initial":
                    from agent.initial_mode import CONCEPT_PRIOR_TEMPLATE_INITIAL
                    prior_template = CONCEPT_PRIOR_TEMPLATE_INITIAL
                concept_prior = scorer.score_image(
                    entry["image_path"], top_k=top_k, template=prior_template,
                )

            prior_report = ""
            prior_image_path = ""
            prior_study = entry.get("prior_study")
            if prior_study and isinstance(prior_study, dict):
                prior_report = prior_study.get("report", "")
                prior_image_path = prior_study.get("image_path", "")

            # Phase 1: initial tool calls (reports + classification)
            from agent.react_agent import AgentTrajectory
            trajectory = AgentTrajectory(
                image_id=study_id, concept_prior=concept_prior,
            )
            system_prompt = agent._build_system_prompt(concept_prior)
            messages = [{
                "role": "user",
                "content": agent._build_initial_message(
                    entry["image_path"],
                    prior_report=prior_report,
                    prior_image_path=prior_image_path,
                ),
            }]

            start_time = time.time()
            agent._react_loop(
                messages, system_prompt, trajectory,
                max_iterations=checkpoint_after,
                force_report_on_max=False,
            )

            # Extract tool evidence gathered so far
            evidence = extract_tool_evidence({"steps": trajectory.steps})

            # Generate checkpoint critique
            critique, fb_in, fb_out = generate_checkpoint_critique(
                evidence, gt_report, agent.client, args.model,
            )
            trajectory.total_input_tokens += fb_in
            trajectory.total_output_tokens += fb_out

            feedback = (
                "Attending feedback before you write the report: " + critique + "\n"
                "Keep this in mind as you continue your investigation."
            )

            # Phase 2: inject feedback and continue
            trajectory.steps.append({
                "iteration": 0,
                "type": "feedback_injection",
                "text": feedback,
            })
            messages.append({"role": "user", "content": feedback})

            agent._react_loop(
                messages, system_prompt, trajectory,
                max_iterations=remaining_iterations,
            )

            trajectory.total_duration_ms = (time.time() - start_time) * 1000
            trajectory.messages = messages
            trajectory.system_prompt = system_prompt

            report = trajectory.final_report
            in_tok = trajectory.total_input_tokens
            out_tok = trajectory.total_output_tokens
            steps = trajectory.steps

        except Exception as e:
            logger.error(f"Checkpoint failed for {study_id}: {e}", exc_info=True)
            report = ""
            in_tok = out_tok = 0
            steps = []
            errors += 1

        wall_ms = (time.time() - t0) * 1000
        cum_in += in_tok
        cum_out += out_tok

        clean_report, groundings = strip_groundings(report)

        pred = {
            "study_id": study_id,
            "report_pred": clean_report,
            "report_pred_raw": report,
            "groundings": groundings,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "num_steps": len(steps),
            "wall_time_ms": wall_ms,
            "base_model": base_name,
            "feedback": feedback,
            "checkpoint_after": checkpoint_after,
        }
        predictions.append(pred)
        existing[study_id] = pred

        traj_record = {
            "study_id": study_id,
            "concept_prior": concept_prior if scorer else "",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "num_steps": len(steps),
            "wall_time_ms": wall_ms,
            "groundings": groundings,
            "feedback": feedback,
            "checkpoint_after": checkpoint_after,
            "steps": steps,
        }
        with open(trajectories_path, "a") as f:
            f.write(json.dumps(traj_record) + "\n")

        if len(predictions) % 5 == 0:
            _save_predictions(predictions_path, predictions)
            done = len(predictions)
            elapsed = time.time() - t_start
            logger.info(
                f"[{done}/{total}] saved | "
                f"tokens: {cum_in:,} in / {cum_out:,} out | "
                f"{elapsed/60:.1f}min elapsed"
            )

    _save_predictions(predictions_path, predictions)
    print(f"\nRITL checkpoint: {len(predictions)} predictions -> {predictions_path}")
    print(f"Trajectories: {trajectories_path}")
    print(f"Total tokens: {cum_in:,} input, {cum_out:,} output")
    if errors:
        print(f"Errors: {errors}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Radiologist-in-the-Loop (RITL) Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Exp 1: text-only revision (no servers needed)
  python scripts/eval_ritl.py --mode text --output results/eval_v4/

  # Exp 1 with 5 samples for debugging
  python scripts/eval_ritl.py --mode text --output results/eval_v4/ --max_samples 5

  # Exp 2: agent re-run with feedback (needs servers)
  python scripts/eval_ritl.py --mode rerun --output results/eval_v4/

  # Score results
  python scripts/eval_mimic.py --mode score --output results/eval_v4/
""",
    )

    parser.add_argument(
        "--mode", required=True, choices=["text", "rerun", "checkpoint"],
        help="text: single Claude revision call | rerun: full agent + feedback | checkpoint: mid-loop feedback",
    )
    parser.add_argument(
        "--output", default="results/eval_v4/",
        help="Output directory (default: results/eval_v4/)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude model for text revision (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--feedback_mode", default="structured", choices=["structured", "full_gt"],
        help="structured (default): directional attending critique | full_gt: verbatim GT (ceiling/training data)",
    )
    parser.add_argument(
        "--predictions_base", default="agent_initial",
        help="Baseline predictions to revise (default: agent_initial)",
    )
    parser.add_argument(
        "--config", default="configs/config_initial.yaml",
        help="Agent config YAML (for rerun mode, default: config_initial.yaml to match eval_v4)",
    )
    parser.add_argument(
        "--max_samples", type=int,
        help="Limit to first N studies (for debugging)",
    )
    parser.add_argument(
        "--max_feedback_iterations", type=int, default=5,
        help="Max agent iterations after feedback injection (rerun mode, default: 5)",
    )
    parser.add_argument(
        "--checkpoint_after", type=int, default=2,
        help="Pause after N iterations for checkpoint mode (default: 2)",
    )

    # For _load_test_set compatibility
    parser.add_argument("--input", help="Path to eval data JSON (overrides test_set.json)")
    parser.add_argument("--track", choices=["baseline", "followup"],
                        help="Filter studies by eval_track")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "text":
        run_text_revision(args)
    elif args.mode == "rerun":
        run_agent_rerun(args)
    elif args.mode == "checkpoint":
        run_checkpoint(args)


if __name__ == "__main__":
    main()
