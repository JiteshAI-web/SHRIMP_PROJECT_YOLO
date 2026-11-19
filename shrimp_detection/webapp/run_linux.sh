#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/jiteshai-web/.local/share/shrimp_venv/bin/python"

echo "========================================"
echo "Shrimp Detection Web Dashboard (Linux)"
echo "========================================"
cd "$SCRIPT_DIR"
exec "$VENV_PYTHON" app.py
