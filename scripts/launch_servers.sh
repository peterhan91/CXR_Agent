#!/bin/bash
#
# Launch CXR model servers for config_combo_full.yaml.
#
# GPU allocation (3x GPU setup):
#   GPU 0: CheXagent-2 (report, srrg, classify, vqa, temporal)     ~18 GB
#   GPU 1: CheXOne + BiomedParse                                    ~18 GB
#   GPU 2: MedGemma (vqa, grounding) + FactCheXcker                 ~12 GB
#
# Usage:
#   bash scripts/launch_servers.sh              # launch all 5 servers
#   bash scripts/launch_servers.sh --only core  # CheXagent-2 + CheXOne only
#   bash scripts/launch_servers.sh --stop       # stop all servers

set -e

# Configuration — adjust paths to your setup
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"

BIOMEDPARSE_DIR="${PARENT_DIR}/BiomedParse-v1"  # v1 worktree (2D CXR); main branch is v2 (3D only)
FACTCHEXCKER_DIR="${PARENT_DIR}/FactCheXcker"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Conda environments — models with incompatible transformers versions get separate envs
CONDA_BASE="$(conda info --base)"
CHEXAGENT2_PYTHON="${CONDA_BASE}/envs/cxr_chexagent2/bin/python"  # transformers==4.40.0
MAIN_PYTHON="${CONDA_BASE}/envs/cxr_agent/bin/python"              # transformers>=4.57

# Stop all servers
if [ "$1" = "--stop" ]; then
    echo "Stopping all servers..."
    pkill -f "chexagent2_server" 2>/dev/null || true
    pkill -f "chexone_server" 2>/dev/null || true
    pkill -f "biomedparse_server" 2>/dev/null || true
    pkill -f "medgemma_server" 2>/dev/null || true
    pkill -f "factchexcker_server" 2>/dev/null || true
    pkill -f "whisper_server" 2>/dev/null || true
    echo "All servers stopped."
    exit 0
fi

echo "=== Launching CXR Model Servers (config_combo_full) ==="
echo "Logs: ${LOG_DIR}/"
echo ""

# --- GPU 0: CheXagent-2 (multi-task) ---
echo "[GPU 0] Starting CheXagent-2 server on :8001 (report, srrg, classify, vqa, temporal)..."
CUDA_VISIBLE_DEVICES=0 "${CHEXAGENT2_PYTHON}" "${REPO_ROOT}/servers/chexagent2_server.py" \
    --port 8001 \
    > "${LOG_DIR}/chexagent2.log" 2>&1 &
echo "  PID: $!"

if [ "$1" = "--only" ] && [ "$2" = "core" ]; then
    # --- GPU 1: CheXOne only ---
    echo "[GPU 1] Starting CheXOne server on :8002..."
    CUDA_VISIBLE_DEVICES=1 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/chexone_server.py" \
        --port 8002 \
        > "${LOG_DIR}/chexone.log" 2>&1 &
    echo "  PID: $!"

    echo ""
    echo "Core servers launched. Use 'bash scripts/launch_servers.sh --stop' to stop."
    exit 0
fi

# --- GPU 1: CheXOne + BiomedParse ---
echo "[GPU 1] Starting CheXOne server on :8002..."
CUDA_VISIBLE_DEVICES=1 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/chexone_server.py" \
    --port 8002 \
    > "${LOG_DIR}/chexone.log" 2>&1 &
echo "  PID: $!"

if [ -d "${BIOMEDPARSE_DIR}" ]; then
    echo "[GPU 1] Starting BiomedParse server on :8005..."
    LD_LIBRARY_PATH="${CONDA_BASE}/envs/cxr_agent/lib:${LD_LIBRARY_PATH}" \
    CUDA_VISIBLE_DEVICES=1 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/biomedparse_server.py" \
        --port 8005 \
        --biomedparse_dir "${BIOMEDPARSE_DIR}" \
        > "${LOG_DIR}/biomedparse.log" 2>&1 &
    echo "  PID: $!"
else
    echo "[SKIP] BiomedParse not found at ${BIOMEDPARSE_DIR}"
fi

# --- GPU 2: MedGemma + FactCheXcker ---
echo "[GPU 2] Starting MedGemma server on :8010 (vqa, grounding)..."
CUDA_VISIBLE_DEVICES=2 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/medgemma_server.py" \
    --port 8010 \
    > "${LOG_DIR}/medgemma.log" 2>&1 &
echo "  PID: $!"

echo "[GPU 2] Starting FactCheXcker server on :8007..."
CUDA_VISIBLE_DEVICES=2 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/factchexcker_server.py" \
    --port 8007 \
    --factchexcker_dir "${FACTCHEXCKER_DIR}" \
    > "${LOG_DIR}/factchexcker.log" 2>&1 &
echo "  PID: $!"

# --- GPU 1: Whisper (voice input for RITL) ---
echo "[GPU 1] Starting Whisper server on :8011 (voice transcription)..."
CUDA_VISIBLE_DEVICES=1 "${MAIN_PYTHON}" "${REPO_ROOT}/servers/whisper_server.py" \
    --port 8011 \
    > "${LOG_DIR}/whisper.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "=== All servers launching (wait 2-3 min for models to load) ==="
echo "Monitor logs: tail -f ${LOG_DIR}/*.log"
echo ""
echo "Health checks:"
echo "  curl http://localhost:8001/health  # CheXagent-2"
echo "  curl http://localhost:8002/health  # CheXOne"
echo "  curl http://localhost:8005/health  # BiomedParse"
echo "  curl http://localhost:8007/health  # FactCheXcker"
echo "  curl http://localhost:8010/health  # MedGemma"
echo "  curl http://localhost:8011/health  # Whisper"
echo ""
echo "Stop all: bash scripts/launch_servers.sh --stop"
