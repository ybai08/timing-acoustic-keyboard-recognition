#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_setup.py

echo
echo "Environment ready."
echo "To use it later, run:"
echo "  source .venv/bin/activate"

