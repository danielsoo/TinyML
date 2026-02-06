#!/usr/bin/env bash
# Vast.ai: single data path (project's data/raw/CIC-IDS2017)
# One full rsync is enough. Uses config/federated_vast.yaml.
#
# First time: pip install -r requirements.txt (flwr etc.)

set -euo pipefail

CONFIG="${1:-config/federated_vast.yaml}"
echo "Using config: $CONFIG (data path: data/raw/CIC-IDS2017)"
echo ""

cd "$(dirname "$0")/.."
python run.py --config "$CONFIG"
