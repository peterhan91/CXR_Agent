#!/usr/bin/env python3
"""Step 1.5: Test key tools on CheXpert-Plus, ReXGradient, IU-Xray images."""

import requests, json, time, sys

DATASETS = {
    "chexpert_plus": "/home/than/Datasets/stanford_mit_chest/CheXpert-v1.0/valid/patient64545/study1/view1_frontal.jpg",
    "rexgradient": "/home/than/.cache/huggingface/hub/datasets--rajpurkarlab--ReXGradient-160K/snapshots/e0c7dd5940c6e5f77aac20eea3ff93825d7f8ff3/images/deid_png/GRDN4SZYTXK18ZTE/GRDNXJ9IBYKTA6ZJ/studies/1.2.826.0.1.3680043.8.498.41432772122085702344820063914742446917/series/1.2.826.0.1.3680043.8.498.31237460284934991919963506231562557994/instances/1.2.826.0.1.3680043.8.498.83442985880054869626823907747404816134.png",
    "iu_xray": "/home/than/Datasets/IU_XRay/images/images_normalized/3030_IM-1405-3001.dcm.png",
}

all_results = {}

for ds_name, img_path in DATASETS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {ds_name}", flush=True)
    print(f"Image: {img_path}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    def test(name, fn):
        t0 = time.time()
        try:
            out = fn()
            dt = time.time() - t0
            results[name] = {"status": "ok", "output": out, "time_s": round(dt, 2)}
            print(f"  OK   {name} ({dt:.1f}s)", flush=True)
        except Exception as e:
            dt = time.time() - t0
            results[name] = {"status": "error", "error": str(e), "time_s": round(dt, 2)}
            print(f"  FAIL {name}: {e}", flush=True)

    # Core report generators
    test("chexagent2_report", lambda: requests.post("http://localhost:8001/generate_report", json={"image_path": img_path}, timeout=120).json())
    test("chexone_report", lambda: requests.post("http://localhost:8002/generate_report", json={"image_path": img_path, "reasoning": False}, timeout=120).json())
    test("medgemma_report", lambda: requests.post("http://localhost:8010/generate_report", json={"image_path": img_path}, timeout=120).json())

    # Classifiers
    test("chexzero_classify", lambda: requests.post("http://localhost:8009/classify", json={"image_path": img_path}, timeout=60).json())
    test("cxr_foundation_classify", lambda: requests.post("http://localhost:8008/classify", json={"image_path": img_path}, timeout=120).json())

    # VQA
    test("chexagent2_vqa", lambda: requests.post("http://localhost:8001/vqa", json={"image_path": img_path, "question": "Describe the main findings in this chest X-ray."}, timeout=60).json())

    # Grounding
    test("biomedparse_segment", lambda: requests.post("http://localhost:8005/segment", json={"image_path": img_path, "prompts": ["left lung", "right lung"]}, timeout=120).json())

    # Grounding phrase
    test("chexagent2_grounding", lambda: requests.post("http://localhost:8001/ground", json={"image_path": img_path, "task": "phrase_grounding", "phrase": "abnormality"}, timeout=120).json())

    all_results[ds_name] = {"image_path": img_path, "results": results}

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n  {ds_name}: {ok}/{len(results)} OK", flush=True)

# Save
with open("results/tool_test_other_datasets.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nAll results saved to results/tool_test_other_datasets.json", flush=True)
