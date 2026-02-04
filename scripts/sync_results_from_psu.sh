#!/bin/bash
# PSU server -> Local: pull only data/processed (analysis, runs)
# Usage: bash scripts/sync_results_from_psu.sh

set -e
cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"
SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== PSU server -> Local (results only) ==="
echo "Server: $SERVER:$REMOTE"
echo "Local:  $LOCAL_ROOT"
echo ""

mkdir -p "$LOCAL_ROOT/data/processed"
rsync -avz --progress "$SERVER:$REMOTE/data/processed/" "$LOCAL_ROOT/data/processed/"

echo ""
echo "Done. Check: ls -la data/processed/analysis/v11/"
echo ""
