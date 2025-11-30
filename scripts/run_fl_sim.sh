#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
# 로컬 환경용 설정 파일 사용
python -m src.federated.client --config config/federated_local.yaml
