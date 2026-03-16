#!/usr/bin/env python3
"""Step 1.4: Test all tools on a single MIMIC-CXR image."""

import requests, json, time, traceback, sys

IMG = "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10046166/s50051329/abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg"
results = {}

def test(name, fn):
    sys.stdout.flush()
    t0 = time.time()
    try:
        out = fn()
        dt = time.time() - t0
        results[name] = {"status": "ok", "output": out, "time_s": round(dt, 2)}
        print(f"  OK   {name} ({dt:.1f}s)", flush=True)
    except Exception as e:
        dt = time.time() - t0
        results[name] = {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "time_s": round(dt, 2)}
        print(f"  FAIL {name}: {e}", flush=True)

# === REPORT GENERATORS ===
print("=== REPORT GENERATORS ===", flush=True)
test("chexagent2_report", lambda: requests.post("http://localhost:8001/generate_report", json={"image_path": IMG}, timeout=120).json())
test("chexagent2_srrg_report", lambda: requests.post("http://localhost:8001/generate_srrg", json={"image_path": IMG}, timeout=120).json())
test("chexone_report", lambda: requests.post("http://localhost:8002/generate_report", json={"image_path": IMG, "reasoning": False}, timeout=120).json())
test("chexone_report_reasoning", lambda: requests.post("http://localhost:8002/generate_report", json={"image_path": IMG, "reasoning": True}, timeout=120).json())
test("medgemma_report", lambda: requests.post("http://localhost:8010/generate_report", json={"image_path": IMG}, timeout=120).json())
test("medversa_report", lambda: requests.post("http://localhost:8004/generate_report", json={"image_path": IMG, "context": "Indication: Chest X-ray.\nComparison: None.", "prompt": "Write a radiology report for <img0> with FINDINGS and IMPRESSION sections.", "modality": "cxr"}, timeout=180).json())

# === CLASSIFIERS ===
print("\n=== CLASSIFIERS ===", flush=True)
test("chexagent2_classify_disease_id", lambda: requests.post("http://localhost:8001/classify", json={"image_path": IMG, "task": "disease_id", "disease_names": ["pneumonia", "atelectasis", "cardiomegaly", "pleural effusion", "pneumothorax", "edema", "consolidation"]}, timeout=60).json())
test("chexagent2_classify_view", lambda: requests.post("http://localhost:8001/classify", json={"image_path": IMG, "task": "view"}, timeout=60).json())
test("chexzero_classify", lambda: requests.post("http://localhost:8009/classify", json={"image_path": IMG}, timeout=60).json())
test("cxr_foundation_classify", lambda: requests.post("http://localhost:8008/classify", json={"image_path": IMG}, timeout=120).json())
test("medversa_classify", lambda: requests.post("http://localhost:8004/classify", json={"image_path": IMG}, timeout=120).json())

# === VQA ===
print("\n=== VQA ===", flush=True)
test("chexagent2_vqa_effusion", lambda: requests.post("http://localhost:8001/vqa", json={"image_path": IMG, "question": "Is there a pleural effusion?"}, timeout=60).json())
test("chexagent2_vqa_heart", lambda: requests.post("http://localhost:8001/vqa", json={"image_path": IMG, "question": "Is the heart enlarged?"}, timeout=60).json())
test("chexagent2_vqa_devices", lambda: requests.post("http://localhost:8001/vqa", json={"image_path": IMG, "question": "What devices or lines are present?"}, timeout=60).json())
test("medgemma_vqa_effusion", lambda: requests.post("http://localhost:8010/vqa", json={"image_path": IMG, "question": "Is there a pleural effusion?"}, timeout=120).json())
test("medgemma_vqa_devices", lambda: requests.post("http://localhost:8010/vqa", json={"image_path": IMG, "question": "What devices or lines are present?"}, timeout=120).json())
test("medversa_vqa_effusion", lambda: requests.post("http://localhost:8004/vqa", json={"image_path": IMG, "question": "Is there a pleural effusion?"}, timeout=120).json())

# === GROUNDING ===
print("\n=== GROUNDING ===", flush=True)
test("chexagent2_grounding_phrase", lambda: requests.post("http://localhost:8001/ground", json={"image_path": IMG, "task": "phrase_grounding", "phrase": "pleural effusion"}, timeout=120).json())
test("chexagent2_grounding_abnormality", lambda: requests.post("http://localhost:8001/ground", json={"image_path": IMG, "task": "abnormality", "disease_name": "cardiomegaly"}, timeout=120).json())
test("chexagent2_grounding_foreign", lambda: requests.post("http://localhost:8001/ground", json={"image_path": IMG, "task": "foreign_objects"}, timeout=120).json())
test("biomedparse_segment", lambda: requests.post("http://localhost:8005/segment", json={"image_path": IMG, "prompts": ["left lung", "right lung", "lung opacity"]}, timeout=120).json())
test("medversa_detect", lambda: requests.post("http://localhost:8004/detect", json={"image_path": IMG}, timeout=120).json())
test("medversa_segment", lambda: requests.post("http://localhost:8004/segment_2d", json={"image_path": IMG}, timeout=180).json())

# === VERIFICATION ===
print("\n=== VERIFICATION ===", flush=True)
test_report = "FINDINGS: The ET tube tip is approximately 2 cm above the carina. The heart is normal in size. The lungs are clear.\nIMPRESSION: ET tube in satisfactory position."
test("factchexcker_verify", lambda: requests.post("http://localhost:8007/verify_report", json={"image_path": IMG, "report": test_report}, timeout=180).json())

# === EVIDENCE BOARD (local, no server) ===
print("\n=== EVIDENCE BOARD ===", flush=True)
sys.path.insert(0, "/home/than/DeepLearning/CXR_Agent")
from tools.evidence_board import EvidenceBoardTool
eb = EvidenceBoardTool()
test("evidence_board_add", lambda: eb.run("add", "cardiomegaly", ["chexzero: present", "cxr_foundation: present"], "bbox=[0.25, 0.30, 0.75, 0.82]"))
test("evidence_board_add2", lambda: eb.run("add", "pleural effusion", ["chexagent2_report: mentioned"]))
test("evidence_board_reject", lambda: eb.run("reject", "pneumonia", reason="only 1 source positive, 3 negative"))
test("evidence_board_list", lambda: eb.run("list"))

# === Save all results ===
with open("results/tool_test_outputs.json", "w") as f:
    json.dump({"image_path": IMG, "dataset": "mimic_cxr", "results": results}, f, indent=2, default=str)

ok = sum(1 for r in results.values() if r["status"] == "ok")
print(f"\nDone! {ok}/{len(results)} tools OK", flush=True)
print(f"Results saved to results/tool_test_outputs.json", flush=True)
