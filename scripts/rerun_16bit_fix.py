#!/usr/bin/env python3
"""
Re-run eval pipeline after 16-bit fix.
Chains all phases sequentially. Each mode skips already-completed studies.

Usage:
    python scripts/rerun_16bit_fix.py [--start-phase N]
"""
import subprocess
import sys
import time
import argparse

PY = "/home/than/anaconda3/envs/cxr_agent/bin/python"
CFG = "configs/config_grounded.yaml"
INPUT = "data/eval/sample_100.json"

PHASES = [
    # Phase 1: Baseline
    ("Phase 1a", f"{PY} scripts/eval_mimic.py --mode agent   --input {INPUT} --track baseline  --output results/eval_baseline/         --config {CFG}"),
    ("Phase 1b", f"{PY} scripts/eval_mimic.py --mode chexone --input {INPUT} --track baseline  --output results/eval_baseline/"),
    ("Phase 1c", f"{PY} scripts/eval_mimic.py --mode medversa --input {INPUT} --track baseline --output results/eval_baseline/"),
    # Phase 2: Followup
    ("Phase 2a", f"{PY} scripts/eval_mimic.py --mode agent   --input {INPUT} --track followup  --output results/eval_followup/         --config {CFG}"),
    ("Phase 2b", f"{PY} scripts/eval_mimic.py --mode chexone --input {INPUT} --track followup  --output results/eval_followup/"),
    ("Phase 2c", f"{PY} scripts/eval_mimic.py --mode medversa --input {INPUT} --track followup --output results/eval_followup/"),
    # Phase 3: CLEAR ablation
    ("Phase 3a", f"{PY} scripts/eval_mimic.py --mode agent   --input {INPUT} --track baseline  --output results/eval_baseline_noclear/  --config {CFG} --no_clear"),
    ("Phase 3b", f"{PY} scripts/eval_mimic.py --mode agent   --input {INPUT} --track followup  --output results/eval_followup_noclear/ --config {CFG} --no_clear"),
    # Phase 4: Score + compare
    ("Phase 4a", f"{PY} scripts/eval_mimic.py --mode score   --output results/eval_baseline/"),
    ("Phase 4b", f"{PY} scripts/eval_mimic.py --mode score   --output results/eval_followup/"),
    ("Phase 4c", f"{PY} scripts/eval_mimic.py --mode score   --output results/eval_baseline_noclear/"),
    ("Phase 4d", f"{PY} scripts/eval_mimic.py --mode score   --output results/eval_followup_noclear/"),
    ("Phase 4e", f"{PY} scripts/eval_mimic.py --mode compare --output results/eval_baseline/"),
    ("Phase 4f", f"{PY} scripts/eval_mimic.py --mode compare --output results/eval_followup/"),
]


def run_phase(name, cmd):
    print(f"\n{'='*60}")
    print(f"  {name}: {cmd.split('--mode')[1].split('--')[0].strip() if '--mode' in cmd else ''}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n", flush=True)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n*** {name} FAILED (exit code {result.returncode}) ***")
        return False

    print(f"\n  {name} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Phase number to start from (1-4)")
    args = parser.parse_args()

    # Map phase number to index
    phase_starts = {1: 0, 2: 3, 3: 6, 4: 8}
    start_idx = phase_starts.get(args.start_phase, 0)

    print(f"Starting from phase {args.start_phase} (step {start_idx + 1}/{len(PHASES)})")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    failed = []
    for i, (name, cmd) in enumerate(PHASES):
        if i < start_idx:
            print(f"Skipping {name}")
            continue

        ok = run_phase(name, cmd)
        if not ok:
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"  ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    else:
        print(f"  All phases succeeded!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
