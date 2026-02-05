#!/usr/bin/env bash
# PSU e5-cse-135-01 server run script
# Avoid home disk quota: redirect temp/cache to /scratch

set -euo pipefail

USER="${USER:-yqp5187}"
export TMPDIR="/scratch/${USER}/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MPLCONFIGDIR="/scratch/${USER}/.config"
export XDG_CONFIG_HOME="/scratch/${USER}/.config"

mkdir -p "$TMPDIR" "$MPLCONFIGDIR" 2>/dev/null || true

# OOM 방지: 이전 run에서 남은 Ray 워커 정리 (메모리 해제)
if command -v ray >/dev/null 2>&1; then
  ray stop --force 2>/dev/null || true
  echo "Ray stopped (cleaned leftover workers)"
fi

# 메모리 사용 가능량 확인 (시작 전)
echo "=== Memory (before run) ==="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "free not available"
fi
echo ""

CONFIG="${1:-config/federated_scratch.yaml}"
echo "Using config: $CONFIG"
echo "TMPDIR=$TMPDIR (avoids home quota)"
echo ""

cd "$(dirname "$0")/.."
python run.py --config "$CONFIG"
