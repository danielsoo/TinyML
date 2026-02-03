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

CONFIG="${1:-config/federated_scratch.yaml}"
echo "Using config: $CONFIG"
echo "TMPDIR=$TMPDIR (avoids home quota)"
echo ""

cd "$(dirname "$0")/.."
python run.py --config "$CONFIG"
