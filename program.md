# CXR Agent

Autonomous experimentation for CXR report generation. The agent reads this file, runs experiments, and iterates — no human in the loop.

## Goal

**Get the lowest RadCliQ-v1 and highest RadGraph-F1 / SembScore / BERTScore / BLEU-2 on MIMIC-CXR test set.** Beat ALL baselines (Sonnet API, CheXOne standalone) on ALL 5 metrics. Use every available tool to get there.

| Metric | Direction | Measures |
|--------|-----------|----------|
| RadCliQ-v1 | lower is better | Composite clinical quality |
| RadGraph-F1 | higher is better | Clinical entity/relation overlap |
| SembScore | higher is better | Sentence embedding similarity |
| BERTScore | higher is better | Contextual embedding F1 |
| BLEU-2 | higher is better | Bigram precision |

Per study, output: report text (FINDINGS + IMPRESSION), grounded findings (JSON with bboxes), two figures (`{id}_bbox.png`, `{id}_mask.png`).

## Tools (9 active)

All validated 2026-03-11. Use as many as possible per study (target 8-12 calls).

| Tool | Server | What it does |
|------|--------|-------------|
| `chexagent2_report` | :8001 | Free-text report |
| `chexagent2_srrg_report` | :8001 | Structured report by anatomy |
| `chexagent2_grounding` | :8001 | Bbox per finding (`task=phrase_grounding`, `phrase="..."`) |
| `chexagent2_classify` | :8001 | Binary disease classification (`task=binary_disease`, `disease_name="..."`) |
| `chexagent2_vqa` | :8001 | Follow-up questions |
| `chexone_report` | :8002 | Second-opinion report (Qwen2.5-VL) |
| `chexzero_classify` | :8008 | Zero-shot 14-label screening (P>0.5 = positive; best for cardiomegaly/edema) |
| `biomedparse_segment` | :8005 | Text-prompted segmentation (`prompts=["left lung"]`; good for anatomy, not pathology) |
| `factchexcker_verify` | :8007 | Report hallucination checker |

Plus **CLEAR concept prior** — DINOv2+CLIP cosine similarity to MIMIC-CXR concepts, injected before tool calls.

**Disabled**: MedVersa (hallucinating), MedSAM (poor CXR masks), MedSAM3 (replaced).

## What you CAN modify

- `agent/prompts.py` — system prompt, templates, skill injection. **Primary lever.**
- `agent/react_agent.py` — ReAct loop, iteration count, tool selection strategy.
- `configs/config_grounded.yaml` — tool enablement, temperature, max_iterations.
- `skills/*.md` — clinical reasoning skills injected into the system prompt.

## What you CANNOT modify

- `scripts/eval_mimic.py` — evaluation harness. Ground truth scorer.
- `tools/*.py` and `servers/*.py` — tool and server implementations.
- `clear/` — CLEAR concept scorer.

## Running experiments

```bash
MIMIC=/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0
CFG=configs/config_grounded.yaml

# Prepare test set (run once)
python scripts/eval_mimic.py --mode prepare --mimic_dir $MIMIC --output results/eval/ --max_samples 5

# Baselines (run once)
python scripts/eval_mimic.py --mode sonnet --output results/eval/ --config $CFG
python scripts/eval_mimic.py --mode chexone --output results/eval/ --config $CFG

# Agent run
python scripts/eval_mimic.py --mode agent --output results/eval_iter_N/ --config $CFG

# Score and compare
python scripts/eval_mimic.py --mode score --output results/eval_iter_N/
python scripts/eval_mimic.py --mode compare --output results/eval_iter_N/
```

Extract key metrics: `cat results/eval_iter_N/comparison.txt`

## Logging results

Log every experiment to `results.tsv` (tab-separated). Do NOT commit this file.

```
commit	radcliq_v1	radgraph_f1	semb_score	bertscore_f1	bleu_2	status	description
—	—	—	—	—	—	baseline	sonnet api vision-only
—	—	—	—	—	—	baseline	chexone direct
a1b2c3d	—	—	—	—	—	keep	initial agent run
```

## The experiment loop

LOOP FOREVER:

1. Read current state: `cat results.tsv`, check which metrics still lag baselines.
2. Hypothesize a targeted change. One idea per iteration. Examples:
   - "Reports too verbose → tighten word count in system prompt"
   - "Low RadGraph-F1 → agent missing entities → add more classification calls"
   - "Low BLEU-2 → wording diverges from radiology conventions → add style examples"
   - "Agent not using enough tools → add explicit instructions to call all 9"
3. Implement: edit `agent/prompts.py`, `skills/*.md`, `configs/config_grounded.yaml`, or `agent/react_agent.py`.
4. git commit.
5. Run: `python scripts/eval_mimic.py --mode agent --output results/eval_iter_N/ --config configs/config_grounded.yaml`
6. Score: `python scripts/eval_mimic.py --mode score --output results/eval_iter_N/`
7. Read results: `cat results/eval_iter_N/comparison.txt`
8. Record in `results.tsv`.
9. If metrics improved → **keep** the commit, advance.
10. If metrics regressed → **discard**, `git reset --hard HEAD~1`.
11. Repeat.

**EvoTest** (Approach B): If manual iteration stalls, use automated skill evolution from `../mimic_skills/EvoTest/`. Train on 20-30 validation CXRs (`results/eval_train/`), test on 5 test CXRs. UCB tree search + Evolver LLM generates improved skills in `skills/`. See `../mimic_skills/` for the full framework.

**Crashes**: If a run crashes, check the error. If it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, log `crash` in results.tsv, revert, and move on.

**Timeout**: Each agent run on 5 CXRs should take ~5-15 minutes. If it exceeds 30 minutes, kill and discard.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human anything. Do NOT ask "should I continue?" or "is this good enough?". The human may be asleep. Continue indefinitely until manually interrupted or all 5 metrics beat all baselines. If you run out of ideas, think harder — re-read tool outputs, try combining approaches, try radical prompt changes, try different skill strategies. The loop runs until the human stops you.

## Reference

- **MIMIC-CXR-JPG**: `/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/`
- **Config**: `configs/config_grounded.yaml`
- **Skill file**: `skills/grounded_report.md`
- **GPU 0**: CheXagent-2 (18.6GB) | **GPU 1**: CheXOne + CheXzero + BiomedParse | **GPU 2**: FactCheXcker + eval
- **Conda envs**: `cxr_agent` (main), `cxr_chexagent2`, `radgraph` (eval step 1), `green_score` (eval steps 2-3)
- **Server safety**: shared GPU server. Never delete outside `CXR_Agent/`, never touch other envs, never kill others' processes.
