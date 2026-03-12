#!/usr/bin/env python3
"""
EvoTest-style skill evolution for CXR report generation.

Evolves the agent's clinical reasoning skill by optimizing 1/RadCliQ-v1
(the ReXrank primary ranking metric) on MIMIC-CXR validation studies,
following the UCB tree-based exploration from mimic_skills.

Computes all 7 ReXrank metrics when available:
  BLEU-2, BERTScore, SembScore, RadGraph, 1/RadCliQ-v1, RaTEScore, GREEN

Usage:
    # Train: evolve skill on 30 val patients, 10 episodes
    python scripts/evotest_cxr.py --mode train --episodes 10 \
        --val-json results/eval_enriched/val_studies_enriched.json \
        --config configs/config.yaml

    # Test: evaluate best evolved skill on 100 test patients
    python scripts/evotest_cxr.py --mode test \
        --test-json results/eval_enriched/test_studies_enriched.json

    # Full: train + test
    python scripts/evotest_cxr.py --mode full --episodes 10 \
        --val-json results/eval_enriched/val_studies_enriched.json \
        --test-json results/eval_enriched/test_studies_enriched.json

    # Resume from checkpoint
    python scripts/evotest_cxr.py --mode train --resume --episodes 15

    # Dry run (prints Evolver prompt, no API calls or agent runs)
    python scripts/evotest_cxr.py --mode train --dry-run --episodes 1

Requires: ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("evotest_cxr")

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ─── Logging ────────────────────────────────────────────────────────────────


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "evotest_cxr.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return str(log_path)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ─── Report Parsing (from eval_mimic.py) ────────────────────────────────────


def strip_groundings(report_text):
    """Strip GROUNDINGS section and preamble from agent report."""
    clean = report_text
    groundings = []
    m = re.search(r'\n*GROUNDINGS:\s*', clean, re.IGNORECASE)
    if m:
        before = clean[:m.start()].strip()
        after = clean[m.end():].strip()
        try:
            groundings = json.loads(after)
        except json.JSONDecodeError:
            bracket_match = re.search(r'\[.*\]', after, re.DOTALL)
            if bracket_match:
                try:
                    groundings = json.loads(bracket_match.group())
                except json.JSONDecodeError:
                    pass
        clean = before

    findings_match = re.search(r'(?:^|\n)\s*FINDINGS?:\s*', clean, re.IGNORECASE)
    if findings_match:
        clean = clean[findings_match.start():].strip()

    clean = re.sub(r'^#+\s+', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
    clean = re.sub(r'---+', '', clean)
    return clean.strip(), groundings


# ─── Sampling ───────────────────────────────────────────────────────────────


def sample_studies(enriched_path, n_subjects, seed=42):
    """Sample n unique patients, pick one study per patient (latest by date).

    Returns list of enriched study entries.
    """
    import random

    with open(enriched_path) as f:
        data = json.load(f)

    # Group by subject_id, pick latest study per subject
    by_subject = {}
    for entry in data:
        sid = entry["subject_id"]
        if sid not in by_subject or entry["study_date"] > by_subject[sid]["study_date"]:
            by_subject[sid] = entry

    subjects = list(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    selected = subjects[:n_subjects]
    studies = [by_subject[s] for s in selected]
    logger.info(f"Sampled {len(studies)} studies from {len(subjects)} subjects (seed={seed})")
    return studies


# ─── Scoring ────────────────────────────────────────────────────────────────

# Lazy-loaded scorer instances (persist across episodes to avoid reloading)
_ratescore_model = None
_green_model = None


def _get_ratescore():
    """Lazy-init RaTEScore model."""
    global _ratescore_model
    if _ratescore_model is None:
        from RaTEScore import RaTEScore
        _ratescore_model = RaTEScore()
        logger.info("  RaTEScore model loaded")
    return _ratescore_model


def _get_green():
    """Lazy-init GREEN evaluator (7B LLM, needs GPU)."""
    global _green_model
    if _green_model is None:
        from green_score.green import GREEN
        _green_model = GREEN(
            model_name="StanfordAIMI/GREEN-radllama2-7b",
            output_dir="/tmp/green_eval",
            cpu=False,
            compute_summary_stats=False,
        )
        logger.info("  GREEN model loaded")
    return _green_model


def _write_report_csv(reports, path):
    """Write reports to CSV in CXR-Report-Metric format."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "report"])
        for i, r in enumerate(reports):
            w.writerow([i, r])


def compute_scores(gt_reports, pred_reports, skip_ratescore=False, skip_green=False):
    """Compute all available ReXrank metrics.

    Tries CXR-Report-Metric (BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ-v1),
    then RaTEScore, then GREEN. Falls back to basic BERTScore+BLEU-2 if
    CXR-Report-Metric is unavailable.

    Returns (per_case_list, agg_dict).
    """
    import tempfile

    n = len(gt_reports)
    per_case = [{} for _ in range(n)]
    agg = {
        "n_studies": n,
        "n_empty": sum(1 for p in pred_reports if not p.strip()),
    }

    # ── CXR-Report-Metric: 5 core metrics ──
    try:
        from CXRMetric.run_eval import calc_metric
        import pandas as pd

        gt_csv = tempfile.mktemp(suffix="_gt.csv")
        pred_csv = tempfile.mktemp(suffix="_pred.csv")
        out_csv = tempfile.mktemp(suffix="_scores.csv")
        _write_report_csv(gt_reports, gt_csv)
        _write_report_csv(pred_reports, pred_csv)

        calc_metric(gt_csv, pred_csv, out_csv, use_idf=False)
        scores_df = pd.read_csv(out_csv)

        col_map = {
            "bleu_score": "bleu_2",
            "bertscore": "bertscore_f1",
            "semb_score": "semb_score",
            "radgraph_combined": "radgraph_f1",
            "RadCliQ-v1": "radcliq_v1",
        }
        for csv_col, key in col_map.items():
            if csv_col in scores_df.columns:
                vals = scores_df[csv_col].tolist()
                agg[key] = sum(vals) / n if n else 0
                for i, v in enumerate(vals):
                    if i < n:
                        per_case[i][key] = float(v)

        # 1/RadCliQ-v1 (higher = better, used on ReXrank leaderboard)
        if "radcliq_v1" in agg and agg["radcliq_v1"] > 0:
            agg["inv_radcliq_v1"] = 1.0 / agg["radcliq_v1"]

        os.unlink(gt_csv)
        os.unlink(pred_csv)
        os.unlink(out_csv)
        rq = f"{agg['radcliq_v1']:.4f}" if "radcliq_v1" in agg else "N/A"
        irq = f"{agg['inv_radcliq_v1']:.4f}" if "inv_radcliq_v1" in agg else "N/A"
        logger.info(f"  CXR-Report-Metric: RadCliQ-v1={rq}, 1/RadCliQ-v1={irq}")
    except (ImportError, Exception) as e:
        logger.warning(f"  CXR-Report-Metric not available ({e}), falling back to basic metrics")
        # Fallback: basic BERTScore + BLEU-2
        from bert_score import score as bert_score_fn
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        _, _, F1 = bert_score_fn(pred_reports, gt_reports, lang="en", verbose=False)
        per_bert = F1.tolist()
        agg["bertscore_f1"] = sum(per_bert) / n if n else 0

        smooth = SmoothingFunction().method1
        per_bleu = []
        for gt, pred in zip(gt_reports, pred_reports):
            if not pred.strip():
                per_bleu.append(0.0)
            else:
                per_bleu.append(sentence_bleu(
                    [gt.split()], pred.split(),
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smooth,
                ))
        agg["bleu_2"] = sum(per_bleu) / n if n else 0

        for i in range(n):
            per_case[i]["bertscore_f1"] = per_bert[i]
            per_case[i]["bleu_2"] = per_bleu[i]

    # ── RaTEScore ──
    if not skip_ratescore:
        try:
            scorer = _get_ratescore()
            scores = scorer.compute_score(
                [str(p) if p.strip() else "No findings." for p in pred_reports],
                [str(g) for g in gt_reports],
            )
            agg["ratescore"] = sum(scores) / n if n else 0
            for i, s in enumerate(scores):
                per_case[i]["ratescore"] = float(s)
            logger.info(f"  RaTEScore: {agg['ratescore']:.4f}")
        except (ImportError, Exception) as e:
            logger.warning(f"  RaTEScore not available: {e}")

    # ── GREEN ──
    if not skip_green:
        try:
            evaluator = _get_green()
            mean, std, green_scores, summary, results_df = evaluator(
                [str(g) for g in gt_reports],
                [str(p) if p.strip() else "No findings." for p in pred_reports],
            )
            if "green_score" in results_df.columns:
                vals = results_df["green_score"].tolist()
                agg["green_score"] = sum(vals) / n if n else 0
                for i, v in enumerate(vals):
                    if i < n:
                        per_case[i]["green_score"] = float(v)
            gs = f"{agg['green_score']:.4f}" if "green_score" in agg else "N/A"
            logger.info(f"  GREEN: {gs}")
        except (ImportError, Exception) as e:
            logger.warning(f"  GREEN not available: {e}")

    return per_case, agg


def composite_score(agg):
    """UCB composite score (higher = better).

    Primary: 1/RadCliQ-v1 (inverse of RadCliQ-v1, the ReXrank ranking metric).
    Fallback: BERTScore F1 if CXR-Report-Metric is unavailable.
    """
    if "inv_radcliq_v1" in agg:
        return agg["inv_radcliq_v1"]
    if "radcliq_v1" in agg and agg["radcliq_v1"] > 0:
        return 1.0 / agg["radcliq_v1"]
    return agg.get("bertscore_f1", 0)


# ─── Agent Execution ────────────────────────────────────────────────────────


def build_clinical_context(admission_info):
    """Build clinical context string from admission info."""
    parts = []
    cc = admission_info.get("chief_complaint", "")
    if cc:
        parts.append(f"Chief Complaint: {cc}")
    hpi = admission_info.get("patient_history", "")
    if hpi:
        parts.append(f"History of Present Illness: {hpi}")
    return "\n\n".join(parts)


def run_agent_on_studies(studies, agent, scorer, config, args):
    """Run the agent on a list of studies. Returns list of prediction dicts."""
    predictions = []
    total = len(studies)

    for i, entry in enumerate(studies):
        study_id = entry["study_id"]
        logger.info(f"  [{i+1}/{total}] study {study_id}")
        t0 = time.time()

        # CLEAR concept prior
        concept_prior = ""
        if scorer:
            top_k = config.get("clear", {}).get("top_k", 20)
            concept_prior = scorer.score_image(entry["image_path"], top_k=top_k)

        # Optional context
        prior_report = ""
        prior_image_path = ""
        clinical_context = ""
        if args.use_prior:
            priors = entry.get("prior_studies", [])
            if priors:
                prior_report = priors[0].get("report", "")
                prior_image_path = priors[0].get("image_path", "")
        if args.use_clinical_context and entry.get("admission_info"):
            clinical_context = build_clinical_context(entry["admission_info"])

        try:
            trajectory = agent.run(
                image_path=entry["image_path"],
                concept_prior_text=concept_prior,
                image_id=study_id,
                prior_report=prior_report,
                prior_image_path=prior_image_path,
                clinical_context=clinical_context,
            )
            report = trajectory.final_report
            clean_report, groundings = strip_groundings(report)
            steps = trajectory.steps
            in_tok = trajectory.total_input_tokens
            out_tok = trajectory.total_output_tokens
        except Exception as e:
            logger.error(f"  Agent failed for {study_id}: {e}")
            clean_report = ""
            groundings = []
            steps = []
            in_tok = out_tok = 0

        wall_ms = (time.time() - t0) * 1000
        predictions.append({
            "study_id": study_id,
            "report_gt": entry["report_gt"],
            "report_pred": clean_report,
            "groundings": groundings,
            "steps": steps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "wall_time_ms": wall_ms,
        })

    return predictions


# ─── Evolver ────────────────────────────────────────────────────────────────


def format_trajectory_summary(pred):
    """Format a single prediction's trajectory for the Evolver."""
    lines = [f"### Study {pred['study_id']}"]

    # Tool calls
    tool_calls = [s for s in pred.get("steps", []) if s.get("type") == "tool_call"]
    if tool_calls:
        tools_used = [f"{s['tool_name']}({json.dumps(s.get('tool_input', {}))[:100]})" for s in tool_calls]
        lines.append(f"**Tools called** ({len(tool_calls)}): {', '.join(t.split('(')[0] for t in tools_used)}")
    else:
        lines.append("**Tools called**: none")

    return "\n".join(lines)


def build_evolver_prompt(parent_node, predictions, per_case_scores, agg_scores,
                         evolution_history="", n_worst=8):
    """Build the Evolver prompt for CXR skill evolution."""

    # --- Aggregate metrics ---
    metrics_lines = []
    # Show all 7 ReXrank metrics in display order
    metric_display = [
        ("bleu_2", "BLEU-2"),
        ("bertscore_f1", "BERTScore"),
        ("semb_score", "SembScore"),
        ("radgraph_f1", "RadGraph"),
        ("inv_radcliq_v1", "1/RadCliQ-v1"),
        ("ratescore", "RaTEScore"),
        ("green_score", "GREEN"),
    ]
    for key, label in metric_display:
        if key in agg_scores:
            metrics_lines.append(f"| {label} | {agg_scores[key]:.4f} |")
    metrics_table = "| Metric | Value |\n|--------|-------|\n" + "\n".join(metrics_lines)

    # --- Parent skill ---
    parent_skill_section = ""
    if parent_node and parent_node.get("skill_text"):
        parent_skill_section = (
            f"## Parent Skill (composite={parent_node['score']:.4f})\n\n"
            f"Analyze where it helped and where it failed, then IMPROVE it:\n\n"
            f"```markdown\n{parent_node['skill_text']}\n```\n\n"
        )

    # --- Worst cases ---
    # Rank by RadCliQ-v1 descending (higher = worse) or BERTScore ascending
    indexed = list(zip(range(len(predictions)), predictions, per_case_scores))
    if any("radcliq_v1" in s for s in per_case_scores):
        indexed.sort(key=lambda x: x[2].get("radcliq_v1", float("inf")), reverse=True)
        sort_label = "RadCliQ-v1, descending"
    else:
        indexed.sort(key=lambda x: x[2].get("bertscore_f1", 0))
        sort_label = "BERTScore F1, ascending"
    worst = indexed[:n_worst]

    worst_cases = []
    for idx, pred, scores in worst:
        # Build per-case metric line
        metric_parts = []
        for key, label in [("radcliq_v1", "RadCliQ-v1"), ("bertscore_f1", "BERTScore"),
                           ("bleu_2", "BLEU-2"), ("radgraph_f1", "RadGraph"),
                           ("ratescore", "RaTEScore"), ("green_score", "GREEN")]:
            if key in scores:
                metric_parts.append(f"**{label}**: {scores[key]:.3f}")
        metric_line = " | ".join(metric_parts) if metric_parts else "(no scores)"

        case = f"""---
{format_trajectory_summary(pred)}

{metric_line}

**Ground Truth Report**:
```
{pred['report_gt'][:500]}
```

**Agent's Predicted Report**:
```
{pred['report_pred'][:500] if pred['report_pred'] else '(empty — agent produced no report)'}
```
"""
        worst_cases.append(case)

    worst_section = "\n".join(worst_cases) if worst_cases else "(no cases to analyze)"

    prompt = f"""You are a radiology AI system optimizer. Analyze CXR report generation results and produce an improved clinical reasoning skill for the agent.

The agent generates chest X-ray radiology reports in MIMIC-CXR style. It has 10 specialized CXR tools (report generators, classifiers, segmenters, grounding, fact-checker) and receives CLEAR concept priors. The skill guides the agent's workflow: which tools to call, how to interpret results, and how to write the final report.

## Evolution History

{evolution_history}

## Current Performance ({agg_scores.get('n_studies', 0)} studies)

{metrics_table}

Empty predictions: {agg_scores.get('n_empty', 0)}

{parent_skill_section}## Worst Cases (ranked by {sort_label})

{worst_section}

## Your Task

Generate an improved CXR report generation workflow skill. The skill should:

1. **Guide tool usage** — which of the 10 tools to call, in what order, how to combine their outputs
   - Report tools: `chexagent2_report`, `chexone_report`, `chexagent2_srrg_report`
   - Classifiers: `chexzero_classify`, `cxr_foundation_classify`, `chexagent2_classify`
   - VQA: `chexagent2_vqa`
   - Grounding: `chexagent2_grounding`, `biomedparse_segment`
   - Verification: `factchexcker_verify`
2. **Teach MIMIC-CXR report style** — plain prose, standard phrases, correct structure
3. **Address the failure patterns above** — analyze GT vs predicted reports, fix what's wrong
4. **Include concrete examples** of well-written MIMIC-CXR reports (normal and abnormal)
5. **Set appropriate report length** — FINDINGS typically 30-80 words, IMPRESSION 5-20 words

The output format MUST be:
```
FINDINGS:
<plain text paragraph>

IMPRESSION:
<1-2 sentence summary>

GROUNDINGS:
[{{"finding": "...", "bbox": [...], "tool": "..."}}]
```

Key quality factors (scored by the ReXrank metric suite):
- **1/RadCliQ-v1** (primary): composite of BLEU-2 + BERTScore + SembScore + RadGraph — covers lexical, semantic, and clinical structure
- **RaTEScore**: factual & temporal consistency — penalizes hallucinated or contradicted findings
- **GREEN**: LLM-based clinical error analysis — catches false/missing findings, wrong location/severity
- **Clinical accuracy**: findings match the ground truth (no hallucinations, no missed findings)
- **Standard phrasing**: use established radiology terminology (e.g. "no focal consolidation" not "lungs are clear")
- **Appropriate length**: concise, not verbose — FINDINGS 30-80 words, IMPRESSION 5-20 words
- **Structure**: plain text FINDINGS then IMPRESSION, no markdown formatting

Output ONLY the skill content in markdown format. No preamble or explanation outside the skill itself. The skill should be 400-800 words."""

    return prompt


def sanitize_skill_text(raw_text):
    """Strip markdown fences and preamble from Evolver output.

    Claude often wraps its response in ```markdown ... ``` fences.
    This function extracts the clean skill content.
    """
    text = raw_text.strip()

    # Strip outer markdown fences (```markdown ... ``` or ``` ... ```)
    # Allow trailing text after the closing fence (Claude sometimes adds commentary)
    fence_pattern = re.compile(
        r'^```(?:markdown|md)?\s*\n(.*?)```', re.DOTALL
    )
    m = fence_pattern.match(text)
    if m:
        text = m.group(1).strip()
    else:
        # If still has leading fence (partial match), strip it
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl != -1:
                text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    # Strip any preamble before the first markdown heading
    heading_match = re.search(r'^#\s', text, re.MULTILINE)
    if heading_match and heading_match.start() > 0:
        preamble = text[:heading_match.start()].strip()
        # Only strip if preamble looks like conversational text (not skill content)
        if len(preamble) < 200 and not preamble.startswith("##"):
            text = text[heading_match.start():]

    return text.strip()


def call_evolver(prompt, model="claude-opus-4-6"):
    """Call Anthropic API to generate improved skill."""
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return sanitize_skill_text(raw)


# ─── EvoTest Engine ─────────────────────────────────────────────────────────


def append_episode_jsonl(log_path, episode_num, node_idx, parent_idx,
                         composite, best_score, agg_scores, duration_s):
    record = {
        "episode": episode_num,
        "node": node_idx,
        "parent": parent_idx,
        "composite": round(composite, 4),
        "best": round(best_score, 4),
        "metrics": {k: round(v, 4) for k, v in agg_scores.items() if isinstance(v, float)},
        "duration_s": round(duration_s, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


class CXREvoTest:
    def __init__(self, args):
        self.args = args
        self.nodes = []
        self.best_node_idx = None
        self.best_score = float("-inf")
        self.last_episode_score = None
        self.completed_episodes = 0

        # Paths
        exp_suffix = f"_{args.experiment}" if args.experiment else ""
        self.state_dir = PROJECT_DIR / f"evotest_state{exp_suffix}"
        self.skills_dir = PROJECT_DIR / "skills" / f"evo{exp_suffix}"
        self.state_file = self.state_dir / "state.json"
        self.episode_log = self.state_dir / "episode_log.jsonl"

        # Agent infrastructure (initialized lazily in run())
        self.agent = None
        self.scorer = None
        self.config = None

    # ── UCB Tree ──

    def calculate_ucb(self, node_idx):
        node = self.nodes[node_idx]
        num_children = len(node["children_idxs"])
        total_nodes = max(2, len(self.nodes))
        c = self.args.exploration_constant
        alpha = self.args.depth_constant
        exploration = c * (alpha ** node["depth"]) * math.sqrt(
            math.log(total_nodes) / (1 + num_children)
        )
        return node["score"] + exploration

    def select_parent(self):
        if not self.nodes:
            return -1

        # Force-best-after-drop safety
        if (
            self.args.force_best_after_drop
            and self.last_episode_score is not None
            and self.best_node_idx is not None
            and (self.best_score - self.last_episode_score) >= self.args.drop_threshold
        ):
            logger.info(
                f"  [UCB] Force-selecting best node {self.best_node_idx} "
                f"(score={self.best_score:.4f}) after drop"
            )
            return self.best_node_idx

        best_ucb_idx = max(range(len(self.nodes)), key=self.calculate_ucb)
        logger.info(
            f"  [UCB] Selected node {best_ucb_idx} "
            f"(score={self.nodes[best_ucb_idx]['score']:.4f}, "
            f"ucb={self.calculate_ucb(best_ucb_idx):.4f})"
        )
        return best_ucb_idx

    # ── Agent Setup ──

    def setup_agent(self):
        """Initialize agent infrastructure (one-time)."""
        from agent.react_agent import CXRReActAgent

        with open(self.args.config) as f:
            self.config = yaml.safe_load(f)

        # CLEAR scorer
        self.scorer = None
        if not self.args.no_clear:
            from clear.concept_scorer import CLEARConceptScorer
            clear_cfg = self.config.get("clear", {})
            self.scorer = CLEARConceptScorer(
                model_path=clear_cfg.get("model_path"),
                concepts_path=clear_cfg.get("concepts_path"),
                dinov2_model_name=clear_cfg.get("dinov2_model_name", "dinov2_vitb14"),
                image_resolution=clear_cfg.get("image_resolution", 448),
            )
            logger.info("Loading CLEAR model...")
            self.scorer.load()
            logger.info("CLEAR model ready")

        # Tools
        tools = self._build_tools()

        # Agent (skill_text will be set per-episode)
        acfg = self.config.get("agent", {})
        self.agent = CXRReActAgent(
            model=acfg.get("model", "claude-sonnet-4-6"),
            max_iterations=acfg.get("max_iterations", 10),
            max_tokens=acfg.get("max_tokens", 4096),
            temperature=acfg.get("temperature", 0.0),
            tools=tools,
            use_skills=False,  # Evolved skill replaces default skills
            reasoning_effort=acfg.get("reasoning_effort"),
        )
        logger.info(f"Agent ready: {len(tools)} tools, model={acfg.get('model', 'claude-sonnet-4-6')}")

    def _build_tools(self):
        from tools import (
            CheXagent2ReportTool, CheXagent2SRRGTool, CheXagent2GroundingTool,
            CheXagent2ClassifyTool, CheXagent2VQATool, CheXOneReportTool,
            CheXzeroClassifyTool, CXRFoundationClassifyTool,
            MedVersaReportTool, MedVersaClassifyTool, MedVersaDetectTool,
            MedVersaSegmentTool, MedVersaVQATool,
            BiomedParseSegmentTool, MedSAMSegmentTool, MedSAM3SegmentTool,
            FactCheXckerVerifyTool,
        )
        tool_config = self.config.get("tools", {})
        registry = {
            "chexagent2": CheXagent2ReportTool,
            "chexagent2_srrg": CheXagent2SRRGTool,
            "chexagent2_classify": CheXagent2ClassifyTool,
            "chexagent2_grounding": CheXagent2GroundingTool,
            "chexagent2_vqa": CheXagent2VQATool,
            "chexone": CheXOneReportTool,
            "chexzero": CheXzeroClassifyTool,
            "cxr_foundation": CXRFoundationClassifyTool,
            "medversa": MedVersaReportTool,
            "medversa_classify": MedVersaClassifyTool,
            "medversa_detect": MedVersaDetectTool,
            "medversa_segment": MedVersaSegmentTool,
            "medversa_vqa": MedVersaVQATool,
            "biomedparse": BiomedParseSegmentTool,
            "medsam": MedSAMSegmentTool,
            "medsam3": MedSAM3SegmentTool,
            "factchexcker": FactCheXckerVerifyTool,
        }
        tools = []
        for key, cls in registry.items():
            entry = tool_config.get(key, {})
            if entry.get("enabled", False):
                endpoint = entry.get("endpoint", "http://localhost:8000")
                tools.append(cls(endpoint=endpoint))
        return tools

    # ── Episode Execution ──

    def run_episode(self, skill_text, episode_num, studies):
        """Run agent with given skill on studies, score, return results."""
        # Save skill file
        if skill_text:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            skill_path = self.skills_dir / f"episode_{episode_num}.md"
            skill_path.write_text(skill_text)
            logger.info(f"  Saved skill: {skill_path} ({len(skill_text)} chars)")

        if self.args.dry_run:
            return 0.0, {}, [], {}

        # Inject skill into agent
        self.agent.skill_text = skill_text

        # Run agent on all studies
        logger.info(f"  Running agent on {len(studies)} studies...")
        predictions = run_agent_on_studies(
            studies, self.agent, self.scorer, self.config, self.args
        )

        # Save predictions
        ep_dir = self.state_dir / f"episode_{episode_num}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        with open(ep_dir / "predictions.json", "w") as f:
            # Strip steps for smaller JSON (keep for worst-case analysis in memory)
            save_preds = [{k: v for k, v in p.items() if k != "steps"} for p in predictions]
            json.dump(save_preds, f, indent=2)

        # Score
        gt_reports = [p["report_gt"] for p in predictions]
        pred_reports = [p["report_pred"] for p in predictions]
        logger.info("  Scoring predictions...")
        per_case, agg = compute_scores(
            gt_reports, pred_reports,
            skip_ratescore=self.args.skip_ratescore,
            skip_green=self.args.skip_green,
        )

        with open(ep_dir / "scores.json", "w") as f:
            json.dump(agg, f, indent=2)

        comp = composite_score(agg)
        return comp, agg, predictions, per_case

    # ── Evolver ──

    def evolve_skill(self, parent_node, predictions, per_case_scores, agg_scores):
        """Call the Evolver to generate an improved skill."""
        # Build evolution history
        history_lines = []
        if self.best_node_idx is not None:
            best = self.nodes[self.best_node_idx]
            history_lines.append(
                f"- **Best skill so far** (node {self.best_node_idx}, "
                f"episode {best['episode_num']}): composite = {best['score']:.4f}"
            )
        if parent_node:
            history_lines.append(
                f"- **Parent skill** (node {parent_node['idx']}, "
                f"episode {parent_node['episode_num']}): composite = {parent_node['score']:.4f}"
            )

        # Recent failed attempts
        failed = [
            n for n in self.nodes
            if n["score"] < (parent_node["score"] if parent_node else 0)
            and n["skill_text"]
        ]
        failed.sort(key=lambda n: n["score"])
        for fn in failed[:3]:
            history_lines.append(
                f"- **Failed attempt** (node {fn['idx']}, "
                f"composite={fn['score']:.4f}): "
                f"skill preview: {fn['skill_text'][:200]}..."
            )

        history = "\n".join(history_lines) if history_lines else "(first episode)"

        prompt = build_evolver_prompt(
            parent_node=parent_node,
            predictions=predictions,
            per_case_scores=per_case_scores,
            agg_scores=agg_scores,
            evolution_history=history,
            n_worst=min(8, len(predictions)),
        )

        if self.args.dry_run:
            logger.info(f"{'='*60}")
            logger.info("DRY RUN — Evolver prompt:")
            logger.info(f"{'='*60}")
            logger.info(prompt[:3000])
            if len(prompt) > 3000:
                logger.info(f"... [{len(prompt) - 3000} chars truncated]")
            return "(dry-run skill placeholder)"

        logger.info(f"  Calling Evolver ({self.args.evolver_model})...")
        skill_text = call_evolver(prompt, model=self.args.evolver_model)
        logger.info(f"  Evolver produced skill ({len(skill_text)} chars)")
        return skill_text

    # ── Persistence ──

    def save_state(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        best_score_safe = self.best_score if math.isfinite(self.best_score) else None
        state = {
            "nodes": list(self.nodes),
            "best_node_idx": self.best_node_idx,
            "best_score": best_score_safe,
            "last_episode_score": self.last_episode_score,
            "completed_episodes": self.completed_episodes,
            "args": {
                "n_train": self.args.n_train,
                "evolver_model": self.args.evolver_model,
                "exploration_constant": self.args.exploration_constant,
                "depth_constant": self.args.depth_constant,
                "drop_threshold": self.args.drop_threshold,
                "force_best_after_drop": self.args.force_best_after_drop,
                "seed": self.args.seed,
            },
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug(f"  State saved ({len(self.nodes)} nodes)")

    def load_state(self):
        if not self.state_file.exists():
            return False
        with open(self.state_file) as f:
            state = json.load(f)
        self.nodes = state["nodes"]
        self.best_node_idx = state["best_node_idx"]
        self.best_score = state["best_score"] if state["best_score"] is not None else float("-inf")
        self.last_episode_score = state["last_episode_score"]
        self.completed_episodes = state["completed_episodes"]
        logger.info(
            f"  Resumed: {self.completed_episodes} episodes, "
            f"{len(self.nodes)} nodes, best={self.best_score:.4f}"
        )
        return True

    # ── Main Loop ──

    def run_train(self, studies):
        """Main evolution loop on training studies."""
        start_time = time.time()
        episode_durations = []

        log_path = setup_logging(self.state_dir)

        if self.args.resume:
            if not self.load_state():
                logger.info("  Cannot resume — starting fresh")

        start_episode = self.completed_episodes
        total_episodes = self.args.episodes

        if start_episode >= total_episodes:
            logger.info(f"Already completed {start_episode} episodes. Increase --episodes.")
            return

        logger.info(f"{'='*70}")
        logger.info(f"EvoTest CXR | {total_episodes} episodes | {len(studies)} studies")
        logger.info(f"{'='*70}")
        logger.info(f"  Evolver:    {self.args.evolver_model}")
        logger.info(f"  UCB c={self.args.exploration_constant}, alpha={self.args.depth_constant}")
        logger.info(f"  State dir:  {self.state_dir}")
        logger.info(f"  Log:        {log_path}")
        logger.info(f"{'='*70}")

        # Load initial skill (grounded_report.md)
        initial_skill_path = self.args.initial_skill or (PROJECT_DIR / "skills" / "grounded_report.md")
        initial_skill = ""
        if Path(initial_skill_path).exists():
            raw = Path(initial_skill_path).read_text()
            # Strip YAML frontmatter
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    initial_skill = parts[2].strip()
                else:
                    initial_skill = raw
            else:
                initial_skill = raw
            logger.info(f"  Initial skill: {initial_skill_path} ({len(initial_skill)} chars)")

        consecutive_failures = 0
        max_consecutive_failures = 3

        for episode_num in range(start_episode, total_episodes):
            ep_start = time.time()
            eta_str = ""
            if episode_durations:
                avg_dur = sum(episode_durations) / len(episode_durations)
                remaining = (total_episodes - episode_num) * avg_dur
                eta_str = f" | ETA: {format_duration(remaining)}"

            logger.info(f"\n{'='*70}")
            logger.info(f"EPISODE {episode_num}/{total_episodes - 1}{eta_str}")
            logger.info(f"{'='*70}")

            try:
                if episode_num == 0 and not self.nodes:
                    # Episode 0: baseline with initial skill
                    skill_text = initial_skill
                    comp, agg, predictions, per_case = self.run_episode(
                        skill_text, episode_num, studies
                    )
                    if not self.args.dry_run and comp == 0.0 and not predictions:
                        logger.error("Episode 0 returned no results — aborting")
                        sys.exit(1)

                    node = {
                        "idx": 0,
                        "skill_text": skill_text,
                        "score": comp,
                        "agg_scores": agg,
                        "parent_idx": -1,
                        "children_idxs": [],
                        "depth": 0,
                        "episode_num": episode_num,
                    }
                    self.nodes.append(node)
                    self.best_node_idx = 0
                    self.best_score = comp
                    self.last_episode_score = comp

                else:
                    # Episodes 1..N: UCB select → evolve → run → score
                    parent_idx = self.select_parent()
                    parent_node = self.nodes[parent_idx]

                    # Load cached parent predictions for Evolver context
                    parent_preds_path = self.state_dir / f"episode_{parent_node['episode_num']}" / "predictions.json"
                    parent_predictions = []
                    parent_per_case = []
                    if parent_preds_path.exists():
                        with open(parent_preds_path) as f:
                            parent_predictions = json.load(f)
                        # Recompute per-case scores (skip expensive metrics — only need ranking)
                        gt = [p["report_gt"] for p in parent_predictions]
                        pred = [p["report_pred"] for p in parent_predictions]
                        parent_per_case, _ = compute_scores(
                            gt, pred, skip_ratescore=True, skip_green=True
                        )

                    # Evolve skill
                    logger.info(f"  Evolving from node {parent_idx} (score={parent_node['score']:.4f})...")
                    new_skill = self.evolve_skill(
                        parent_node, parent_predictions, parent_per_case,
                        parent_node.get("agg_scores", {}),
                    )

                    # Run episode with new skill
                    logger.info(f"  Running episode with evolved skill...")
                    comp, agg, predictions, per_case = self.run_episode(
                        new_skill, episode_num, studies
                    )

                    node = {
                        "idx": len(self.nodes),
                        "skill_text": new_skill,
                        "score": comp,
                        "agg_scores": agg,
                        "parent_idx": parent_idx,
                        "children_idxs": [],
                        "depth": parent_node["depth"] + 1,
                        "episode_num": episode_num,
                    }
                    self.nodes.append(node)
                    parent_node["children_idxs"].append(node["idx"])

                    self.last_episode_score = comp
                    if comp > self.best_score:
                        self.best_score = comp
                        self.best_node_idx = node["idx"]
                        logger.info(f"  *** NEW BEST: composite={comp:.4f} (node {node['idx']}) ***")

                consecutive_failures = 0

            except Exception as e:
                # Graceful failure: log, create failed node, save state, continue
                logger.error(f"  Episode {episode_num} FAILED: {e}", exc_info=True)
                consecutive_failures += 1

                # Episode 0 is the baseline — cannot continue without it
                if episode_num == 0:
                    logger.error("  Baseline episode failed — aborting")
                    sys.exit(1)

                if episode_num > 0:
                    # Create a failed node so the tree remains consistent
                    parent_idx = self.select_parent() if self.nodes else -1
                    node = {
                        "idx": len(self.nodes),
                        "skill_text": "",
                        "score": 0.0,
                        "agg_scores": {},
                        "parent_idx": parent_idx,
                        "children_idxs": [],
                        "depth": (self.nodes[parent_idx]["depth"] + 1) if parent_idx >= 0 else 0,
                        "episode_num": episode_num,
                    }
                    self.nodes.append(node)
                    if parent_idx >= 0:
                        self.nodes[parent_idx]["children_idxs"].append(node["idx"])

                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"  {max_consecutive_failures} consecutive failures — aborting"
                    )
                    self.completed_episodes = episode_num + 1
                    self.save_state()
                    break

                # Save state and continue to next episode
                self.completed_episodes = episode_num + 1
                self.save_state()
                continue

            # Episode summary
            node = self.nodes[-1]
            ep_elapsed = time.time() - ep_start
            episode_durations.append(ep_elapsed)

            logger.info(f"  Episode {episode_num}: composite={node['score']:.4f} | "
                        f"best={self.best_score:.4f} (node {self.best_node_idx}) | "
                        f"{format_duration(ep_elapsed)}")
            agg = node.get("agg_scores", {})
            for key in ["inv_radcliq_v1", "radcliq_v1", "bertscore_f1", "bleu_2",
                         "semb_score", "radgraph_f1", "ratescore", "green_score"]:
                if key in agg:
                    logger.info(f"    {key}: {agg[key]:.4f}")

            append_episode_jsonl(
                self.episode_log, episode_num, node["idx"],
                node.get("parent_idx", -1), node["score"],
                self.best_score, agg, ep_elapsed,
            )

            self.completed_episodes = episode_num + 1
            self.save_state()

        # Final summary
        total_elapsed = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"EVOTEST COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"  Episodes:   {self.completed_episodes}")
        logger.info(f"  Nodes:      {len(self.nodes)}")
        logger.info(f"  Best score: {self.best_score:.4f} (node {self.best_node_idx})")
        logger.info(f"  Total time: {format_duration(total_elapsed)}")

        if self.best_node_idx is not None:
            best = self.nodes[self.best_node_idx]
            best_skill_path = self.skills_dir / f"episode_{best['episode_num']}.md"
            logger.info(f"  Best skill: {best_skill_path}")

        # Score progression
        if len(self.nodes) > 1:
            logger.info(f"\n  Score progression:")
            max_s = max(n["score"] for n in self.nodes)
            min_s = min(n["score"] for n in self.nodes)
            rng = max_s - min_s if max_s != min_s else 1
            for n in self.nodes:
                bar_len = int(30 * (n["score"] - min_s) / rng)
                bar = "#" * bar_len + "." * (30 - bar_len)
                best_marker = " * BEST" if n["idx"] == self.best_node_idx else ""
                logger.info(f"    Ep {n['episode_num']:>2d}: {bar} {n['score']:.4f}{best_marker}")

    def run_test(self, studies, skill_text=None):
        """Evaluate a skill on test studies."""
        if skill_text is None and self.best_node_idx is not None:
            skill_text = self.nodes[self.best_node_idx]["skill_text"]
        elif skill_text is None:
            # Try loading from state
            if self.state_file.exists():
                self.load_state()
                if self.best_node_idx is not None:
                    skill_text = self.nodes[self.best_node_idx]["skill_text"]

        if not skill_text:
            logger.error("No skill to test. Run --mode train first.")
            sys.exit(1)

        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: Evaluating best skill on {len(studies)} studies")
        logger.info(f"{'='*70}")

        self.agent.skill_text = skill_text

        predictions = run_agent_on_studies(
            studies, self.agent, self.scorer, self.config, self.args
        )

        gt_reports = [p["report_gt"] for p in predictions]
        pred_reports = [p["report_pred"] for p in predictions]
        per_case, agg = compute_scores(
            gt_reports, pred_reports,
            skip_ratescore=self.args.skip_ratescore,
            skip_green=self.args.skip_green,
        )

        # Save test results
        test_dir = self.state_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        save_preds = [{k: v for k, v in p.items() if k != "steps"} for p in predictions]
        with open(test_dir / "predictions.json", "w") as f:
            json.dump(save_preds, f, indent=2)
        with open(test_dir / "scores.json", "w") as f:
            json.dump(agg, f, indent=2)

        logger.info(f"\n{'='*70}")
        logger.info(f"TEST RESULTS ({len(studies)} studies)")
        logger.info(f"{'='*70}")
        for key, val in agg.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")

        comp = composite_score(agg)
        logger.info(f"  composite: {comp:.4f}")
        logger.info(f"  Results saved: {test_dir}")

        return agg


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EvoTest-style skill evolution for CXR report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "test", "full"],
        help="train: evolve skill | test: evaluate best skill | full: train + test",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes for training (default: 10)")
    parser.add_argument("--val-json", default="results/eval_enriched/val_studies_enriched.json",
                        help="Enriched val studies JSON")
    parser.add_argument("--test-json", default="results/eval_enriched/test_studies_enriched.json",
                        help="Enriched test studies JSON")
    parser.add_argument("--n-train", type=int, default=30, help="Unique patients for training (default: 30)")
    parser.add_argument("--n-test", type=int, default=100, help="Unique patients for testing (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--config", default="configs/config.yaml", help="Agent config YAML")

    # Agent options
    parser.add_argument("--no-clear", action="store_true", help="Skip CLEAR concept scoring")
    parser.add_argument("--use-prior", action="store_true", help="Feed prior CXR report to agent")
    parser.add_argument("--use-clinical-context", action="store_true", help="Feed HPI + CC to agent")

    # Scoring options
    parser.add_argument("--skip-ratescore", action="store_true", help="Skip RaTEScore (saves time during training)")
    parser.add_argument("--skip-green", action="store_true", help="Skip GREEN score (saves GPU memory + time)")

    # Evolver options
    parser.add_argument("--evolver-model", default="claude-opus-4-6", help="Model for skill evolution")
    parser.add_argument("--initial-skill", default=None, help="Path to initial seed skill (default: skills/grounded_report.md)")

    # UCB tree options
    parser.add_argument("--exploration-constant", type=float, default=1.0, help="UCB exploration c")
    parser.add_argument("--depth-constant", type=float, default=0.8, help="UCB depth decay alpha")
    parser.add_argument("--drop-threshold", type=float, default=0.3, help="Force-best-after-drop threshold (1/RadCliQ-v1 scale)")
    parser.add_argument("--force-best-after-drop", action="store_true", default=True)
    parser.add_argument("--no-force-best-after-drop", action="store_false", dest="force_best_after_drop")

    # Execution options
    parser.add_argument("--experiment", default="", help="Experiment suffix for isolated runs")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--dry-run", action="store_true", help="Print Evolver prompt, skip agent runs")

    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY required")
        sys.exit(1)

    runner = CXREvoTest(args)

    # Initialize agent (shared across train + test)
    if not args.dry_run:
        runner.setup_agent()

    if args.mode in ("train", "full"):
        val_studies = sample_studies(args.val_json, args.n_train, seed=args.seed)
        runner.run_train(val_studies)

    if args.mode in ("test", "full"):
        test_studies = sample_studies(args.test_json, args.n_test, seed=args.seed)
        runner.run_test(test_studies)


if __name__ == "__main__":
    main()
