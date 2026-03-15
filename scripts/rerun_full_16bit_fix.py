#!/usr/bin/env python3
"""
Re-run agent eval with FULL 16-bit fix (server-side + agent-side).

Server-side: _load_cxr() in all 7 servers (already deployed)
Agent-side:  _encode_image() always normalizes 16-bit PNGs (just fixed)

Only reruns --mode agent (CheXOne/MedVersa are unaffected by agent-side fix).
Skips already-completed studies for resumability.

Usage:
    nohup python scripts/rerun_full_16bit_fix.py > logs/rerun_full_fix.log 2>&1 &
"""
import subprocess
import sys
import time
import argparse

PY = "/home/than/anaconda3/envs/cxr_agent/bin/python"
CFG = "configs/config_grounded.yaml"
INPUT = "data/eval/sample_100.json"

PHASES = [
    # Phase 1: Agent predictions (4 tracks)
    ("Agent baseline",
     f"{PY} scripts/eval_mimic.py --mode agent --input {INPUT} --track baseline "
     f"--output results/eval_baseline/ --config {CFG}"),
    ("Agent followup",
     f"{PY} scripts/eval_mimic.py --mode agent --input {INPUT} --track followup "
     f"--output results/eval_followup/ --config {CFG}"),
    ("Agent baseline (no CLEAR)",
     f"{PY} scripts/eval_mimic.py --mode agent --input {INPUT} --track baseline "
     f"--output results/eval_baseline_noclear/ --config {CFG} --no_clear"),
    ("Agent followup (no CLEAR)",
     f"{PY} scripts/eval_mimic.py --mode agent --input {INPUT} --track followup "
     f"--output results/eval_followup_noclear/ --config {CFG} --no_clear"),
    # Phase 2: Score all tracks
    ("Score baseline",
     f"{PY} scripts/eval_mimic.py --mode score --output results/eval_baseline/"),
    ("Score followup",
     f"{PY} scripts/eval_mimic.py --mode score --output results/eval_followup/"),
    ("Score baseline (no CLEAR)",
     f"{PY} scripts/eval_mimic.py --mode score --output results/eval_baseline_noclear/"),
    ("Score followup (no CLEAR)",
     f"{PY} scripts/eval_mimic.py --mode score --output results/eval_followup_noclear/"),
    # Phase 3: Compare
    ("Compare baseline",
     f"{PY} scripts/eval_mimic.py --mode compare --output results/eval_baseline/"),
    ("Compare followup",
     f"{PY} scripts/eval_mimic.py --mode compare --output results/eval_followup/"),
]


def run_phase(name, cmd):
    print(f"\n{'='*60}")
    print(f"  {name}")
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
                        help="Phase index to start from (1-based)")
    args = parser.parse_args()

    start_idx = max(0, args.start_phase - 1)

    print(f"Full 16-bit fix rerun (server + agent)")
    print(f"Starting from phase {args.start_phase}/{len(PHASES)}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backup: results/eval_server_fix_only/")

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
