#!/bin/bash
#
# Launch all CXR model servers on a 3x A6000 GPU setup.
#
# GPU allocation:
#   GPU 0: CheXagent-2 (multi-task: report, grounding, classify, VQA) + CLEAR (in-process)
#   GPU 1: CheXOne + BiomedParse
#   GPU 2: MedVersa (multi-task: report, classify, detect, segment, VQA) + MedSAM3 + FactCheXcker
#
# Usage:
#   bash scripts/launch_servers.sh              # launch all
#   bash scripts/launch_servers.sh --only core  # CheXagent-2 + CheXOne only
#   bash scripts/launch_servers.sh --stop       # stop all servers

set -e

# Configuration — adjust paths to your setup
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"

MEDVERSA_DIR="${PARENT_DIR}/MedVersa"
BIOMEDPARSE_DIR="${PARENT_DIR}/BiomedParse"
MEDSAM3_DIR="${PARENT_DIR}/MedSAM3"
FACTCHEXCKER_DIR="${PARENT_DIR}/FactCheXcker"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Stop all servers
if [ "$1" = "--stop" ]; then
    echo "Stopping all servers..."
    pkill -f "chexagent2_server" 2>/dev/null || true
    pkill -f "chexone_server" 2>/dev/null || true
    pkill -f "medversa_server" 2>/dev/null || true
    pkill -f "biomedparse_server" 2>/dev/null || true
    pkill -f "medsam3_server" 2>/dev/null || true
    pkill -f "factchexcker_server" 2>/dev/null || true
    echo "All servers stopped."
    exit 0
fi

echo "=== Launching CXR Model Servers ==="
echo "Logs: ${LOG_DIR}/"
echo ""

# --- GPU 0: CheXagent-2 (multi-task) ---
echo "[GPU 0] Starting CheXagent-2 server on :8001 (report, grounding, classify, VQA)..."
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/servers/chexagent2_server.py" \
    --port 8001 \
    > "${LOG_DIR}/chexagent2.log" 2>&1 &
echo "  PID: $!"

if [ "$1" = "--only" ] && [ "$2" = "core" ]; then
    # --- GPU 1: CheXOne only ---
    echo "[GPU 1] Starting CheXOne server on :8002..."
    CUDA_VISIBLE_DEVICES=1 python "${REPO_ROOT}/servers/chexone_server.py" \
        --port 8002 \
        > "${LOG_DIR}/chexone.log" 2>&1 &
    echo "  PID: $!"

    echo ""
    echo "Core servers launched. Use 'bash scripts/launch_servers.sh --stop' to stop."
    exit 0
fi

# --- GPU 1: CheXOne + BiomedParse ---
echo "[GPU 1] Starting CheXOne server on :8002..."
CUDA_VISIBLE_DEVICES=1 python "${REPO_ROOT}/servers/chexone_server.py" \
    --port 8002 \
    > "${LOG_DIR}/chexone.log" 2>&1 &
echo "  PID: $!"

if [ -d "${BIOMEDPARSE_DIR}" ]; then
    echo "[GPU 1] Starting BiomedParse server on :8005..."
    CUDA_VISIBLE_DEVICES=1 python "${REPO_ROOT}/servers/biomedparse_server.py" \
        --port 8005 \
        --biomedparse_dir "${BIOMEDPARSE_DIR}" \
        > "${LOG_DIR}/biomedparse.log" 2>&1 &
    echo "  PID: $!"
else
    echo "[SKIP] BiomedParse not found at ${BIOMEDPARSE_DIR}"
fi

# --- GPU 2: MedVersa (multi-task) + MedSAM3 + FactCheXcker ---
if [ -d "${MEDVERSA_DIR}" ]; then
    echo "[GPU 2] Starting MedVersa server on :8004 (report, classify, detect, segment, VQA)..."
    CUDA_VISIBLE_DEVICES=2 python "${REPO_ROOT}/servers/medversa_server.py" \
        --port 8004 \
        --medversa_dir "${MEDVERSA_DIR}" \
        > "${LOG_DIR}/medversa.log" 2>&1 &
    echo "  PID: $!"
else
    echo "[SKIP] MedVersa not found at ${MEDVERSA_DIR}"
fi

if [ -d "${MEDSAM3_DIR}" ]; then
    echo "[GPU 2] Starting MedSAM3 server on :8006..."
    CUDA_VISIBLE_DEVICES=2 python "${REPO_ROOT}/servers/medsam3_server.py" \
        --port 8006 \
        --medsam3_dir "${MEDSAM3_DIR}" \
        > "${LOG_DIR}/medsam3.log" 2>&1 &
    echo "  PID: $!"
else
    echo "[SKIP] MedSAM3 not found at ${MEDSAM3_DIR}"
fi

echo "[GPU 2] Starting FactCheXcker server on :8007..."
CUDA_VISIBLE_DEVICES=2 python "${REPO_ROOT}/servers/factchexcker_server.py" \
    --port 8007 \
    --factchexcker_dir "${FACTCHEXCKER_DIR}" \
    > "${LOG_DIR}/factchexcker.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "=== All servers launching ==="
echo "Monitor logs: tail -f ${LOG_DIR}/*.log"
echo "Health checks:"
echo "  curl http://localhost:8001/health  # CheXagent-2 (multi-task)"
echo "  curl http://localhost:8002/health  # CheXOne"
echo "  curl http://localhost:8004/health  # MedVersa (multi-task)"
echo "  curl http://localhost:8005/health  # BiomedParse"
echo "  curl http://localhost:8006/health  # MedSAM3"
echo "  curl http://localhost:8007/health  # FactCheXcker"
echo ""
echo "Stop all: bash scripts/launch_servers.sh --stop"
