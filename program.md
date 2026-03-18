# CXR Agent — A/B Test: Initial Agent vs Baselines

4 studies × 4 datasets = 16 cases. Agent (initial mode) vs 2 baselines (CheXOne, MedVersa). Reports only (FINDINGS + IMPRESSION). All 7 ReXrank metrics. Results saved to `results/eval_ab/`.

## Tool Inventory

19 tools across 8 servers + 1 in-process scorer. The initial agent uses 8 of these (CheXagent-2 suite + CheXOne + BiomedParse + FactCheXcker).

| Category | Tool | Server (port) | What it does |
|----------|------|---------------|--------------|
| **Report** | `chexagent2_report` | CheXagent-2 (8001) | Free-text FINDINGS/IMPRESSION via 3B VLM |
| **Report** | `chexagent2_srrg_report` | CheXagent-2 (8001) | Structured report by anatomical region |
| **Report** | `chexone_report` | CheXOne (8002) | Qwen2.5-VL-3B with optional reasoning trace |
| **Report** | `medversa_report` | MedVersa (8004) | LLaMA-2-7B with patient context support |
| **Report** | `medgemma_report` | MedGemma (8010) | Google 4B, markdown-formatted |
| **Classify** | `chexagent2_classify` | CheXagent-2 (8001) | View/binary/multi-disease classification |
| **Classify** | `medversa_classify` | MedVersa (8004) | 33 pathology categories (unreliable) |
| **Classify** | `chexzero_classify` | CheXzero (8009) | 14 CheXpert labels, 10-model ensemble, AUC 0.897 |
| **Classify** | `cxr_foundation_classify` | CXR Foundation (8008) | Google ELIXR v2, conservative, CPU-only |
| **VQA** | `chexagent2_vqa` | CheXagent-2 (8001) | Short-answer targeted questions |
| **VQA** | `medgemma_vqa` | MedGemma (8010) | Verbose paragraph answers |
| **VQA** | `medversa_vqa` | MedVersa (8004) | Unreliable — often truncated |
| **Grounding** | `chexagent2_grounding` | CheXagent-2 (8001) | Bounding boxes for findings/devices |
| **Grounding** | `medversa_detect` | MedVersa (8004) | Abnormality detection (often broken) |
| **Segment** | `biomedparse_segment` | BiomedParse (8005) | Text-prompted, verified CXR prompts |
| **Segment** | `medversa_segment` | MedVersa (8004) | 2D segmentation, Dice 0.955 on organs |
| **Segment** | `medsam_segment` | MedSAM (8009) | Requires bbox input (cascade from grounding) |
| **Segment** | `medsam3_segment` | MedSAM3 (8006) | Text-guided, broad vocabulary |
| **Verify** | `factchexcker_verify` | FactCheXcker (8007) | ETT measurement correction |

**CLEAR concept prior** (DINOv2+CLIP, 368K concepts) runs in-process, no server.

## Step 1: Sample 4 studies per dataset

Filter: GT must have **both** FINDINGS and IMPRESSION sections (non-empty).

```bash
TODAY=$(date +%Y%m%d)
OUT=results/eval_ab
SAMPLE=data/eval/sample_4

python3 -c "
import json, os, random
random.seed(42)

datasets = {
    'mimic_cxr': 'data/eval/mimic_cxr_test.json',
    'chexpert_plus': 'data/eval/chexpert_plus_valid.json',
    'rexgradient': 'data/eval/rexgradient_test.json',
    'iu_xray': 'data/eval/iu_xray_test.json',
}

def has_both_sections(study):
    f = study.get('findings', '').strip()
    i = study.get('impression', '').strip()
    return bool(f) and bool(i)

os.makedirs('$SAMPLE', exist_ok=True)
all_samples = []
for name, path in datasets.items():
    d = json.load(open(path))
    studies = d if isinstance(d, list) else d.get('baseline', d.get('studies', []))
    eligible = [s for s in studies if has_both_sections(s)]
    print(f'{name}: {len(eligible)}/{len(studies)} eligible')
    sample = random.sample(eligible, min(4, len(eligible)))
    with open(f'$SAMPLE/{name}_4.json', 'w') as f:
        json.dump(sample, f, indent=2)
    all_samples.extend(sample)
    print(f'  -> {len(sample)} sampled')

with open('$SAMPLE/all_16.json', 'w') as f:
    json.dump(all_samples, f, indent=2)
print(f'Total: {len(all_samples)} studies in $SAMPLE/all_16.json')
"
```

## Step 2: Verify servers + API

Only CheXagent-2 (8001), CheXOne (8002), BiomedParse (8005), FactCheXcker (8007) needed for initial agent. MedVersa (8004) needed for baseline.

```bash
# Required servers for this eval
FAIL=0
for port in 8001 8002 8004 8005 8007; do
  STATUS=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:$port/health)
  if [ "$STATUS" = "200" ]; then echo "Port $port: OK"
  else echo "Port $port: FAIL (HTTP $STATUS)"; FAIL=1; fi
done
[ "$FAIL" = "0" ] || { echo "BLOCKED: restart failed servers"; exit 1; }

# Anthropic API
conda run -n cxr_agent python3 -c "
import anthropic
c = anthropic.Anthropic()
r = c.messages.create(model='claude-sonnet-4-6', max_tokens=10, messages=[{'role':'user','content':'Say OK'}])
print(f'Anthropic API: OK ({r.model})')
"

# ReXrank scoring envs
cd /home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric && \
  conda run -n radgraph python -c "from CXRMetric.run_eval import calc_metric; print('CXR-Report-Metric: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent
conda run -n green_score python -c "from RaTEScore import RaTEScore; print('RaTEScore: OK')"
cd /home/than/DeepLearning/GREEN && \
  conda run -n green_score python -c "from green_score.green import GREEN; print('GREEN: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent
```

## Step 3: Run 2 baselines (direct model calls, no agent)

Each baseline calls its server endpoint directly. No CLEAR, no agent orchestration.

```bash
OUT=results/eval_ab
INPUT=data/eval/sample_4/all_16.json

# CheXOne baseline
conda run -n cxr_agent python scripts/eval_mimic.py --mode chexone \
  --input "$INPUT" --output "$OUT/"

# MedVersa baseline
conda run -n cxr_agent python scripts/eval_mimic.py --mode medversa \
  --input "$INPUT" --output "$OUT/"
```

## Step 4: Run agent (initial mode)

Uses `config_initial.yaml`: original system prompt, original tool descriptions, 8 tools, max 10 iterations, top-20 CLEAR concepts, no skills.

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode agent \
  --input "$INPUT" --output "$OUT/" \
  --config configs/config_initial.yaml
```

Expected output files:
```
results/eval_ab/
├── test_set.json                       # 16 studies (copied from input)
├── predictions_chexone.json            # 16 CheXOne reports
├── predictions_medversa.json           # 16 MedVersa reports
├── predictions_agent_initial.json      # 16 initial-agent reports
└── trajectories_agent_initial.jsonl    # Full agent trajectories
```

## Step 5: Export CSVs for scoring

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode score --output "$OUT/"
```

Creates `scores/{model}/{section}/gt*.csv + pred*.csv` for all 3 methods × 2 sections (findings, reports) × per-dataset + overall. We only care about `reports/` (FINDINGS + IMPRESSION) for this test.

## Step 6: Run ReXrank scoring (7 metrics)

```bash
bash scripts/score_rexrank.sh "$OUT/"
```

Computes for every CSV pair:
1. **CXR-Report-Metric** (radgraph env): BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ-v1
2. **RaTEScore** (green_score env): CXR-specific entity matching
3. **GREEN** (green_score env): StanfordAIMI/GREEN-radllama2-7b

Output per CSV pair: `scores_{tag}.json` with all 7 metrics merged.

## Step 7: View results

```bash
# Summary table (auto-generated by score_rexrank.sh)
cat "$OUT/scores/summary.txt"

# Or via eval_mimic.py compare mode
conda run -n cxr_agent python scripts/eval_mimic.py --mode compare --output "$OUT/"
```

## Step 8: Monitor agent trajectories

```bash
python3 -c "
import json

# Read trajectories (has full step details)
trajs = {}
with open('results/eval_ab/trajectories_agent_initial.jsonl') as f:
    for line in f:
        t = json.loads(line)
        trajs[t['study_id']] = t

# Read predictions (has timing + token counts)
preds = json.load(open('results/eval_ab/predictions_agent_initial.json'))
for p in preds:
    sid = p['study_id']
    t = trajs.get(sid, {})
    tool_calls = [s for s in t.get('steps', []) if s.get('type') == 'tool_call']
    ds = sid.split('_')[0]
    print(f'{ds}/{sid}: {len(tool_calls)} tools, '
          f'{p.get(\"input_tokens\",0)} in, {p.get(\"wall_time_ms\",0)/1000:.1f}s')

print(f'Total: {len(preds)} studies')
all_tools = [len([s for s in trajs.get(p['study_id'],{}).get('steps',[]) if s.get('type')=='tool_call']) for p in preds]
avg_steps = sum(all_tools) / len(preds)
avg_time = sum(p.get('wall_time_ms',0) for p in preds) / len(preds) / 1000
print(f'Avg: {avg_steps:.1f} tool calls, {avg_time:.1f}s per study')
"
```

## Expected results structure

```
results/eval_ab/
├── test_set.json
├── predictions_chexone.json
├── predictions_medversa.json
├── predictions_agent_initial.json
├── trajectories_agent_initial.jsonl
└── scores/
    ├── summary.json                    # All metrics aggregated
    ├── summary.txt                     # Human-readable table
    ├── agent_initial/
    │   └── reports/
    │       ├── gt.csv / pred.csv                     (16 studies)
    │       ├── gt_mimic_cxr.csv / pred_mimic_cxr.csv (4 studies)
    │       ├── gt_chexpert_plus.csv / pred_chexpert_plus.csv
    │       ├── gt_rexgradient.csv / pred_rexgradient.csv
    │       ├── gt_iu_xray.csv / pred_iu_xray.csv
    │       ├── scores_overall.json     (7 metrics)
    │       ├── scores_mimic_cxr.json
    │       ├── scores_chexpert_plus.json
    │       ├── scores_rexgradient.json
    │       └── scores_iu_xray.json
    ├── chexone/
    │   └── reports/
    │       └── ... (same structure)
    └── medversa/
        └── reports/
            └── ... (same structure)
```

---

## Setup reference

| Port | Server | GPU | Tools |
|------|--------|-----|-------|
| 8001 | CheXagent-2 | 0 | `chexagent2_report`, `chexagent2_srrg_report`, `chexagent2_grounding`, `chexagent2_classify`, `chexagent2_vqa` |
| 8002 | CheXOne | 1 | `chexone_report` |
| 8004 | MedVersa | 1 | `medversa_report`, `medversa_classify`, `medversa_detect`, `medversa_segment`, `medversa_vqa` |
| 8005 | BiomedParse | 1 | `biomedparse_segment` |
| 8007 | FactCheXcker | 2 | `factchexcker_verify` |
| 8008 | CXR Foundation | CPU | `cxr_foundation_classify` |
| 8009 | CheXzero | 1 | `chexzero_classify` |
| 8010 | MedGemma | 2 | `medgemma_vqa`, `medgemma_report` |

**CLEAR concept prior** (DINOv2+CLIP) runs in-process, no server.

| Conda env | Purpose |
|-----------|---------|
| `cxr_agent` | Main agent + all servers except CheXagent-2 |
| `cxr_chexagent2` | CheXagent-2 server only (transformers==4.40.0) |
| `radgraph` | CXR-Report-Metric scoring (BLEU, BERT, Semb, RadGraph, RadCliQ) |
| `green_score` | RaTEScore + GREEN scoring |

## Server startup

```bash
# GPU 0: CheXagent-2
conda run -n cxr_chexagent2 python servers/chexagent2_server.py --port 8001

# GPU 1: CheXOne + BiomedParse + MedVersa
CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002
CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005
CUDA_VISIBLE_DEVICES=1 python servers/medversa_server.py --port 8004

# GPU 2: FactCheXcker
OPENAI_API_KEY="..." WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python servers/factchexcker_server.py --port 8007 --factchexcker_dir ../FactCheXcker
```

## A/B test dimensions

What differs between initial mode (`config_initial.yaml`) and current mode (`config_grounded.yaml`):

| Dimension | Initial mode | Current mode |
|-----------|-------------|--------------|
| System prompt | 8-line original (role + hard constraints) | Grounded report prompt + skill workflow |
| Tool descriptions | Original 2-3 lines each | Verbose with WHEN TO USE, EXAMPLE OUTPUT, warnings |
| Tool count | 8 (no MedVersa/MedGemma/CheXzero/CXR-Foundation/MedSAM) | 16+ |
| User message | Verbose "Please analyze..." | Terse "Generate a radiology report..." + image_path |
| CLEAR top_k | 20 concepts | 10 concepts |
| Concept prior wording | "from {N} MIMIC-CXR concepts" | "for this image ({N} shown" |
| Max iterations | 10 | 15 |
| Skills | Disabled | Enabled (grounded_report.md) |
| Reasoning effort | None | "medium" (adaptive thinking) |
| MedVersa payload | `{image_path, context}` only | `+ prompt, modality` fields |
