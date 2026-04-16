#!/bin/bash
# run_local.sh — Build and run benchmark locally (interactive, no Slurm)
# Usage: bash scripts/run_local.sh [BLAS=mkl|openblas|none] [extra args...]
#
# The script also starts the web server so you can watch results in real-time.

set -euo pipefail
cd "$(dirname "$0")/.."

BLAS="${BLAS:-openblas}"
THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_NUM_THREADS="$THREADS"

echo "=== TSMM Local Run ==="
echo "BLAS=$BLAS  OMP_NUM_THREADS=$THREADS"
echo ""

# ── Build ────────────────────────────────────────────────────
make BLAS="$BLAS" -j"$(nproc)"

mkdir -p web

# ── Start web server in background ──────────────────────────
python3 web/server.py &
WEB_PID=$!
echo "Web dashboard: http://localhost:8080  (PID $WEB_PID)"
echo ""

cleanup() {
    echo "Stopping web server (PID $WEB_PID)…"
    kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run benchmark ────────────────────────────────────────────
./benchmark --output web/results.json "$@"
