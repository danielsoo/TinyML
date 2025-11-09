#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python -m src.federated.client
