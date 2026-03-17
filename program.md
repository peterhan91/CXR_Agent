# CXR Agent — v6 Evaluation Program

30 studies × 4 datasets = 120 cases. Agent (no-skill) vs 4 baselines (CheXOne, CheXagent-2, MedVersa, MedGemma). All 7 ReXrank metrics. Results saved to `results/eval_YYYYMMDD/`.

## Step 1: Sample 30 studies per dataset

Filter: GT must have **both** FINDINGS and IMPRESSION sections (non-empty).

Available pool:
- MIMIC-CXR: 1,120 eligible / 1,687 total
- CheXpert-Plus: 62 eligible / 62 total
- ReXGradient: 5,573 eligible / 5,573 total
- IU-Xray: 590 eligible / 590 total

```bash
TODAY=$(date +%Y%m%d)
OUT=results/eval_${TODAY}
SAMPLE=data/eval/sample_30

python3 -c "
import json, os, random, re
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

os.makedirs('data/eval/sample_30', exist_ok=True)
all_samples = []
for name, path in datasets.items():
    d = json.load(open(path))
    studies = d if isinstance(d, list) else d.get('baseline', d.get('studies', []))
    eligible = [s for s in studies if has_both_sections(s)]
    print(f'{name}: {len(eligible)}/{len(studies)} eligible')
    sample = random.sample(eligible, min(30, len(eligible)))
    with open(f'data/eval/sample_30/{name}_30.json', 'w') as f:
        json.dump(sample, f, indent=2)
    all_samples.extend(sample)
    print(f'  -> {len(sample)} sampled')

# Also save combined test_set.json (needed by --mode score)
with open('data/eval/sample_30/all_120.json', 'w') as f:
    json.dump(all_samples, f, indent=2)
print(f'Total: {len(all_samples)} studies in data/eval/sample_30/all_120.json')
"
```

## Step 2: Verify servers + API

```bash
# All 8 servers must be healthy
FAIL=0
for port in 8001 8002 8004 8005 8007 8008 8009 8010; do
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

## Step 3: Run 4 baselines (direct model calls, no agent)

Each baseline calls the server endpoint directly. All use the combined `all_120.json` input and write to a single output directory so scoring sees all 4 datasets together.

```bash
TODAY=$(date +%Y%m%d)
OUT=results/eval_${TODAY}
INPUT=data/eval/sample_30/all_120.json

# CheXOne
conda run -n cxr_agent python scripts/eval_mimic.py --mode chexone \
  --input "$INPUT" --output "$OUT/"

# CheXagent-2
conda run -n cxr_agent python scripts/eval_mimic.py --mode chexagent2 \
  --input "$INPUT" --output "$OUT/"

# MedVersa
conda run -n cxr_agent python scripts/eval_mimic.py --mode medversa \
  --input "$INPUT" --output "$OUT/"

# MedGemma
conda run -n cxr_agent python scripts/eval_mimic.py --mode medgemma \
  --input "$INPUT" --output "$OUT/"
```

## Step 4: Run agent (no-skill, plain prompt)

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode agent \
  --input "$INPUT" --output "$OUT/" \
  --config configs/config_grounded.yaml --no_skills
```

Expected output files:
```
results/eval_YYYYMMDD/
├── test_set.json                       # 120 studies (copied from input)
├── predictions_chexone.json            # 120 CheXOne reports
├── predictions_chexagent2.json         # 120 CheXagent-2 reports
├── predictions_medversa.json           # 120 MedVersa reports
├── predictions_medgemma.json           # 120 MedGemma reports
├── predictions_agent_noskill.json      # 120 agent reports + groundings
└── trajectories_agent_noskill.jsonl    # Full agent trajectories
```

## Step 5: Export CSVs for scoring

```bash
conda run -n cxr_agent python scripts/eval_mimic.py --mode score --output "$OUT/"
```

This creates `scores/{model}/{section}/gt*.csv + pred*.csv` for all 5 methods × 2 sections (findings, reports) × per-dataset + overall.

## Step 6: Run ReXrank scoring (7 metrics)

```bash
bash scripts/score_rexrank.sh "$OUT/"
```

This computes for every CSV pair:
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
preds = json.load(open('$OUT/predictions_agent_noskill.json'))
for p in preds:
    steps = [s for s in p.get('steps', []) if s.get('type') == 'tool_call']
    ds = p['study_id'].split('_')[0]
    print(f'{ds}/{p[\"study_id\"]}: {len(steps)} tools, '
          f'{p.get(\"input_tokens\",0)} in, {p.get(\"wall_time_ms\",0)/1000:.1f}s')
print(f'Total: {len(preds)} studies')
avg_steps = sum(len([s for s in p.get('steps',[]) if s.get('type')=='tool_call']) for p in preds) / len(preds)
avg_time = sum(p.get('wall_time_ms',0) for p in preds) / len(preds) / 1000
print(f'Avg: {avg_steps:.1f} tool calls, {avg_time:.1f}s per study')
"
```

## Expected results structure

```
results/eval_YYYYMMDD/
├── test_set.json
├── predictions_chexone.json
├── predictions_chexagent2.json
├── predictions_medversa.json
├── predictions_medgemma.json
├── predictions_agent_noskill.json
├── trajectories_agent_noskill.jsonl
└── scores/
    ├── summary.json                    # All metrics aggregated
    ├── summary.txt                     # Human-readable table
    ├── agent_noskill/
    │   ├── findings/
    │   │   ├── gt.csv / pred.csv                     (120 studies)
    │   │   ├── gt_mimic_cxr.csv / pred_mimic_cxr.csv (30 studies)
    │   │   ├── gt_chexpert_plus.csv / pred_chexpert_plus.csv
    │   │   ├── gt_rexgradient.csv / pred_rexgradient.csv
    │   │   ├── gt_iu_xray.csv / pred_iu_xray.csv
    │   │   ├── scores_overall.json     (7 metrics)
    │   │   ├── scores_mimic_cxr.json
    │   │   ├── scores_chexpert_plus.json
    │   │   ├── scores_rexgradient.json
    │   │   └── scores_iu_xray.json
    │   └── reports/
    │       └── ... (same structure)
    ├── chexone/
    │   ├── findings/
    │   └── reports/
    ├── chexagent2/
    │   ├── findings/
    │   └── reports/
    ├── medversa/
    │   ├── findings/
    │   └── reports/
    └── medgemma/
        ├── findings/
        └── reports/
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

# GPU 1: CheXOne + BiomedParse + CheXzero + MedVersa
CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002
CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005
CUDA_VISIBLE_DEVICES=1 python servers/chexzero_server.py --port 8009
CUDA_VISIBLE_DEVICES=1 python servers/medversa_server.py --port 8004

# GPU 2: MedGemma + FactCheXcker
CUDA_VISIBLE_DEVICES=2 python servers/medgemma_server.py --port 8010
OPENAI_API_KEY="..." WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python servers/factchexcker_server.py --port 8007 --factchexcker_dir ../FactCheXcker

# CPU: CXR Foundation
python servers/cxr_foundation_server.py --port 8008
```
