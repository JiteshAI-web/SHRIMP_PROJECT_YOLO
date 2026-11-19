#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Shrimp Detection & Monitoring Dashboard (Linux Launcher)"
echo "============================================================"

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

echo "[INFO] Starting Flask web dashboard on http://127.0.0.1:5000 ..."
cd "$SCRIPT_DIR/shrimp_detection/webapp"
exec "$SCRIPT_DIR/venv/bin/python" app.py
