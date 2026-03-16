# CXR Agent — v5.1 Evaluation Program

Three phases: (1) test tools with real CXRs and update descriptions, (2) verify evidence board works, (3) run plain-prompt agent on 5 studies per dataset and compare against baselines.

## Phase 1: Test all tools with real CXRs and update descriptions

### Goal
1. Research each model online to understand its capabilities, known limitations, and intended use
2. Call every enabled tool on 1 real image from each of the 5 datasets
3. Capture real outputs and update tool descriptions with actual examples

### Step 1.1: Research model capabilities online

Before testing, look up each model to understand what it's designed for and how it actually performs. This informs how we describe the tools to Sonnet.

| Model | What to look up |
|-------|----------------|
| **CheXagent-2** | Paper/HuggingFace card — what tasks does it support? Report quality on MIMIC-CXR? Grounding accuracy? Known failure modes? |
| **CheXOne** | Paper/repo — what architecture (Qwen2.5-VL)? Training data? How does reasoning mode differ from instruct? |
| **MedGemma** | Google Health AI blog / model card — what CXR tasks was it validated on? Known biases? |
| **MedVersa** | Paper — what are the 33 pathology categories? Detection accuracy? Segmentation quality on CXR specifically? |
| **CheXzero** | Paper (Tiu et al. 2022) — zero-shot performance on CheXpert? Which pathologies is it strongest/weakest on? |
| **CXR Foundation** | Google ELIXR paper — how does it compare to CheXzero? Complementary or redundant? |
| **BiomedParse** | Paper — which CXR prompts are validated? Does it work on pathology terms beyond the verified list? |
| **FactCheXcker** | Paper/repo — what objects can it verify? Only ETT/carina or broader? False positive rate? |

Use web search to find: model cards, papers, benchmark results, known limitations. Save findings to `results/model_research.md` so we can reference them when writing tool descriptions.

### Step 1.2: Pick test images

Use processed eval datasets at `/home/than/DeepLearning/CXR_Agent/data/eval/`. Pick 1 study from each:

```bash
python3 -c "
import json, os
datasets = {
    'mimic_cxr': 'data/eval/mimic_cxr_test.json',
    'chexpert_plus': 'data/eval/chexpert_plus_valid.json',
    'rexgradient': 'data/eval/rexgradient_test.json',
    'iu_xray': 'data/eval/iu_xray_test.json',
    'padchest_gr': 'data/eval/padchest_gr_test.json',
}
for name, path in datasets.items():
    if os.path.exists(path):
        d = json.load(open(path))
        studies = d if isinstance(d, list) else d.get('baseline', d.get('studies', []))
        if studies:
            print(f'{name}: {studies[0][\"image_path\"]}')
        else:
            print(f'{name}: NO STUDIES')
    else:
        print(f'{name}: FILE MISSING')
"
```

### Step 1.3: Verify servers alive (same as Setup Step 1 below)

### Step 1.4: Call every tool on 1 MIMIC-CXR image

Capture real outputs from all tools:
- All 5 [REPORT GENERATOR] tools
- All 4 [CLASSIFIER] tools
- All 3 [VQA] tools (with realistic questions like "Is there a pleural effusion?", "Is the heart enlarged?", "What devices are present?")
- All 4 [GROUNDING] tools (phrase_grounding for focal findings, biomedparse for diffuse)
- [VERIFICATION] tool (with a report mentioning ETT)
- [MEMORY] evidence_board (add, reject, list)

Save all outputs to `results/tool_test_outputs.json`.

### Step 1.5: Test on other datasets

Repeat tool calls for 1 image from each of: CheXpert-Plus, ReXGradient, IU-Xray, PadChest-GR. Focus on:
- **PadChest-GR**: 16-bit PNG — do tools handle it? Does the agent's normalization work?
- **ReXGradient**: different image resolution — any tool failures?
- **IU-Xray**: different report style — do report generators adapt?
- **CheXpert-Plus**: Stanford data — any path issues?

Log any failures or unexpected outputs.

### Step 1.6: Update tool descriptions with real examples

Using real outputs from step 1.4 + model knowledge from step 1.1:
- Replace synthetic EXAMPLE OUTPUT in `tools/*.py` with actual tool outputs
- Add model-specific notes discovered from research (e.g., "CheXzero is weakest on Fracture and Lung Lesion" or "BiomedParse returns 0% coverage for prompts outside its training set")
- Keep descriptions short: `[GROUP] Model name. What it does. Known strength/weakness. EXAMPLE: <real output>`
- Confirm the [GROUP] tag, WHEN TO USE, and EXAMPLE are all present and accurate

## Phase 2: Verify evidence board works in agent loop

### Goal
Run the agent on 1 study with `--no_skills` and confirm the evidence_board tool is called correctly.

### Steps

```bash
# Run agent on 1 MIMIC-CXR baseline study with plain prompt (no skills)
TEST_IMG=$(python3 -c "
import json
d = json.load(open('data/eval/mimic_cxr_test.json'))
studies = d if isinstance(d, list) else d.get('baseline', d.get('studies', []))
print(studies[0]['image_path'])
")

python scripts/run_agent.py \
  --image "$TEST_IMG" \
  --config configs/config_grounded.yaml \
  --no_skills \
  --output results/phase2_memory_test/
```

### Verify in trajectory
Check `results/phase2_memory_test/*_result.json` trajectory for:
- [ ] Agent calls `evidence_board(action='add', ...)` at least once
- [ ] Agent calls `evidence_board(action='list')` before writing final report
- [ ] Agent calls `evidence_board(action='reject', ...)` for at least one finding
- [ ] Final report only contains findings that appear in the evidence board confirmed list
- [ ] GROUNDINGS section matches grounding info from the evidence board

If evidence_board is not called, check:
- Is it in the tool schemas? (print `agent._tool_schemas` names)
- Is the system prompt mentioning [MEMORY]? (check `SYSTEM_PROMPT_PLAIN`)
- Try adding `"Use the evidence_board tool to track your findings."` to the user message

## Phase 3: Plain-prompt agent vs baselines on 5 studies × 5 datasets

### Goal
Run the plain-prompt agent (no skills, no evolving) on 5 studies from each dataset. Compare against CheXOne, CheXagent-2, and MedVersa baselines. Report all 7 ReXrank metrics.

### Prepare 5-study samples

```bash
# Create 5-study samples from each dataset
python3 -c "
import json, os, random
random.seed(42)

datasets = {
    'mimic_cxr': 'data/eval/mimic_cxr_test.json',
    'chexpert_plus': 'data/eval/chexpert_plus_valid.json',
    'rexgradient': 'data/eval/rexgradient_test.json',
    'iu_xray': 'data/eval/iu_xray_test.json',
    'padchest_gr': 'data/eval/padchest_gr_test.json',
}

os.makedirs('data/eval/sample_5', exist_ok=True)
for name, path in datasets.items():
    if not os.path.exists(path):
        print(f'SKIP {name}: file missing')
        continue
    d = json.load(open(path))
    studies = d if isinstance(d, list) else d.get('baseline', d.get('studies', []))
    sample = random.sample(studies, min(5, len(studies)))
    out = f'data/eval/sample_5/{name}_5.json'
    json.dump(sample, out_f:=open(out, 'w'), indent=2); out_f.close()
    print(f'{name}: {len(sample)} studies -> {out}')
"
```

### Run baselines (direct model calls, no agent)

```bash
OUT=results/eval_v51

for DATASET in mimic_cxr chexpert_plus rexgradient iu_xray padchest_gr; do
  INPUT=data/eval/sample_5/${DATASET}_5.json
  [ -f "$INPUT" ] || continue

  # CheXOne baseline
  python scripts/eval_mimic.py --mode chexone \
    --input "$INPUT" --output "$OUT/$DATASET/"

  # CheXagent-2 baseline
  python scripts/eval_mimic.py --mode chexagent2 \
    --input "$INPUT" --output "$OUT/$DATASET/"

  # MedVersa baseline
  python scripts/eval_mimic.py --mode medversa \
    --input "$INPUT" --output "$OUT/$DATASET/"
done
```

### Run plain-prompt agent

```bash
CFG=configs/config_grounded.yaml

for DATASET in mimic_cxr chexpert_plus rexgradient iu_xray padchest_gr; do
  INPUT=data/eval/sample_5/${DATASET}_5.json
  [ -f "$INPUT" ] || continue

  python scripts/eval_mimic.py --mode agent \
    --input "$INPUT" --output "$OUT/$DATASET/" \
    --config $CFG --no_skills
done
```

### Score all predictions

```bash
for DATASET in mimic_cxr chexpert_plus rexgradient iu_xray padchest_gr; do
  [ -d "$OUT/$DATASET/" ] || continue
  python scripts/eval_mimic.py --mode score --output "$OUT/$DATASET/"
  python scripts/eval_mimic.py --mode compare --output "$OUT/$DATASET/"
done
```

### Expected outputs

```
results/eval_v51/
├── mimic_cxr/
│   ├── predictions_agent.json        # agent trajectories + reports
│   ├── predictions_chexone.json      # CheXOne reports
│   ├── predictions_chexagent2.json   # CheXagent-2 reports
│   ├── predictions_medversa.json     # MedVersa reports
│   └── scores/
│       └── summary.json              # 7 metrics × 4 methods
├── chexpert_plus/
│   └── ...
├── rexgradient/
│   └── ...
├── iu_xray/
│   └── ...
└── padchest_gr/
    └── ...
```

### Monitor agent trajectories

For each agent run, check:
- Number of iterations (should be ≤15)
- Tool calls per iteration (evidence_board usage)
- Total input/output tokens
- Wall time per study
- Final report: does it have FINDINGS + IMPRESSION + GROUNDINGS?

```bash
# Quick trajectory summary
python3 -c "
import json, glob
for f in sorted(glob.glob('results/eval_v51/*/predictions_agent.json')):
    preds = json.load(open(f))
    dataset = f.split('/')[-2]
    for p in preds:
        steps = [s for s in p.get('steps', []) if s.get('type') == 'tool_call']
        eb_calls = [s for s in steps if s.get('tool_name') == 'evidence_board']
        print(f'{dataset}/{p[\"study_id\"]}: {len(steps)} tool calls, {len(eb_calls)} evidence_board, '
              f'{p.get(\"input_tokens\",0)} in_tok, {p.get(\"wall_time_ms\",0)/1000:.1f}s')
"
```

### Print final comparison table

```bash
python3 -c "
import json, glob, os
print(f'{\"Dataset\":<15} {\"Method\":<15} {\"BLEU-2\":>8} {\"BERTSc\":>8} {\"SembSc\":>8} {\"RadGr\":>8} {\"1/RCQ\":>8} {\"RaTE\":>8} {\"GREEN\":>8}')
print('-' * 95)
for dataset_dir in sorted(glob.glob('results/eval_v51/*/')):
    ds = os.path.basename(dataset_dir.rstrip('/'))
    summary_path = os.path.join(dataset_dir, 'scores', 'summary.json')
    if not os.path.exists(summary_path):
        continue
    summary = json.load(open(summary_path))
    for method, scores in summary.items():
        row = [ds, method]
        for k in ['bleu_2', 'bertscore_f1', 'semb_score', 'radgraph_f1', 'inv_radcliq_v1', 'ratescore', 'green_score']:
            row.append(f'{scores.get(k, 0):.4f}')
        print(f'{row[0]:<15} {row[1]:<15} {\"  \".join(row[2:])}')
"
```

---

## Setup (same servers as before)

### Step 1: Verify all servers alive

```bash
FAIL=0
for port in 8001 8002 8004 8005 8007 8008 8009 8010; do
  STATUS=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:$port/health)
  if [ "$STATUS" = "200" ]; then
    echo "Port $port: OK"
  else
    echo "Port $port: FAIL (HTTP $STATUS)"
    FAIL=1
  fi
done
[ "$FAIL" = "0" ] || { echo "BLOCKED: restart failed servers before continuing"; exit 1; }
```

| Port | Server | Tools |
|------|--------|-------|
| 8001 | CheXagent-2 | `chexagent2_report`, `chexagent2_srrg_report`, `chexagent2_grounding`, `chexagent2_classify`, `chexagent2_vqa` |
| 8002 | CheXOne | `chexone_report` |
| 8004 | MedVersa | `medversa_report`, `medversa_classify`, `medversa_detect`, `medversa_segment`, `medversa_vqa` |
| 8005 | BiomedParse | `biomedparse_segment` |
| 8007 | FactCheXcker | `factchexcker_verify` |
| 8008 | CXR Foundation | `cxr_foundation_classify` |
| 8009 | CheXzero | `chexzero_classify` |
| 8010 | MedGemma | `medgemma_vqa`, `medgemma_report` |

Plus **evidence_board** (local, no server) and **CLEAR concept prior** (DINOv2+CLIP).

### Step 2: Verify Anthropic API key

```bash
python3 -c "
import anthropic
c = anthropic.Anthropic()
r = c.messages.create(model='claude-sonnet-4-6', max_tokens=10, messages=[{'role':'user','content':'Say OK'}])
print(f'Anthropic API: OK (model={r.model})')
"
```

### Step 3: Verify ReXrank scoring pipeline

```bash
cd /home/than/DeepLearning/ReXrank-metric/scripts/CXR-Report-Metric && \
  conda run -n radgraph python -c "from CXRMetric.run_eval import calc_metric; print('CXR-Report-Metric: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent

conda run -n green_score python -c "from RaTEScore import RaTEScore; print('RaTEScore: OK')"

cd /home/than/DeepLearning/GREEN && \
  conda run -n green_score python -c "from green_score.green import GREEN; print('GREEN: OK')" && \
  cd /home/than/DeepLearning/CXR_Agent
```

---

## Reference

- **Eval data**: `/home/than/DeepLearning/CXR_Agent/data/eval/`
- **MIMIC-CXR-JPG**: `/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/`
- **Config**: `configs/config_grounded.yaml`
- **Agent mode**: plain prompt (`--no_skills`), evidence_board always enabled
- **GPU 0**: CheXagent-2 (18.6GB) | **GPU 1**: CheXOne + CheXzero + BiomedParse | **GPU 2**: MedGemma + FactCheXcker + eval | **CPU**: CXR Foundation
- **Conda envs**: `cxr_agent` (main), `cxr_chexagent2`, `radgraph` (CXR-Report-Metric), `green_score` (RaTEScore + GREEN)
- **Server safety**: shared GPU server. Never delete outside `CXR_Agent/`, never touch other envs, never kill others' processes.

## Server startup commands

```bash
# GPU 0: CheXagent-2
conda run -n cxr_chexagent2 python servers/chexagent2_server.py --port 8001

# GPU 1: CheXOne + BiomedParse + CheXzero
CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002
CUDA_VISIBLE_DEVICES=1 python servers/biomedparse_server.py --port 8005
CUDA_VISIBLE_DEVICES=1 python servers/chexzero_server.py --port 8009

# GPU 1/2: MedVersa
CUDA_VISIBLE_DEVICES=1 python servers/medversa_server.py --port 8004

# GPU 2: MedGemma + FactCheXcker
CUDA_VISIBLE_DEVICES=2 python servers/medgemma_server.py --port 8010
OPENAI_API_KEY="..." WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python servers/factchexcker_server.py --port 8007 --factchexcker_dir ../FactCheXcker

# CPU: CXR Foundation
python servers/cxr_foundation_server.py --port 8008
```
