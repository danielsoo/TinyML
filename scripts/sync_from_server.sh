#!/usr/bin/env bash
# Sync all TinyML results from PSU server to local TinyML-results.
# Run: cd /path/to/TinyML-results && bash scripts/sync_from_server.sh
# Or:  cd /path/to/Research && bash TinyML-results/scripts/sync_from_server.sh

set -e
SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE_BASE="/scratch/yqp5187/TinyML-main/data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== TinyML-results: sync from PSU server ==="
echo "Server: $SERVER"
echo "Remote: $REMOTE_BASE"
echo "Local:  $LOCAL_BASE"
echo ""

echo "Syncing processed/analysis/ ..."
mkdir -p "$LOCAL_BASE/processed/analysis"
rsync -avz --progress "$SERVER:$REMOTE_BASE/processed/analysis/" "$LOCAL_BASE/processed/analysis/"
echo "Done: processed/analysis"
echo ""

echo "Syncing processed/runs/ ..."
mkdir -p "$LOCAL_BASE/processed/runs"
rsync -avz --progress "$SERVER:$REMOTE_BASE/processed/runs/" "$LOCAL_BASE/processed/runs/"
echo "Done: processed/runs"
echo ""

if ssh -o ConnectTimeout=5 "$SERVER" "[ -d $REMOTE_BASE/models ]" 2>/dev/null; then
  echo "Syncing models/ ..."
  mkdir -p "$LOCAL_BASE/models"
  rsync -avz --progress "$SERVER:$REMOTE_BASE/models/" "$LOCAL_BASE/models/" 2>/dev/null || true
  echo "Done: models"
else
  echo "Skipping models (not on server)"
fi
echo ""

if ssh -o ConnectTimeout=5 "$SERVER" "[ -d $REMOTE_BASE/outputs ]" 2>/dev/null; then
  echo "Syncing outputs/ ..."
  mkdir -p "$LOCAL_BASE/outputs"
  rsync -avz --progress "$SERVER:$REMOTE_BASE/outputs/" "$LOCAL_BASE/outputs/" 2>/dev/null || true
  echo "Done: outputs"
else
  echo "Skipping outputs (not on server)"
fi

echo ""
echo "Sync from server finished."
