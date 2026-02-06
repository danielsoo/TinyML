#!/bin/bash
# Verify local and PSU server files match exactly.
# Usage: bash scripts/verify_sync.sh

set -e
cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== Local vs server sync check ==="
echo "Local:  $LOCAL_ROOT"
echo "Server: $SERVER:$REMOTE"
echo ""

echo "Comparing by checksum..."
OUTPUT=$(rsync -avnc --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/raw' --exclude='data/processed' --exclude='* 2.*' --exclude='* 2' --exclude='TinyML-results' --exclude='outputs' "$SERVER:$REMOTE/" "$LOCAL_ROOT/" 2>/dev/null || true)

DIFF_FILES=$(echo "$OUTPUT" | grep -E '^\./.' | grep -v '^total size' | grep -v '^sent ' | grep -v '^received ' | grep -v 'speedup' || true)

if [ -z "$DIFF_FILES" ]; then
  echo ""
  echo "Local and server match (by checksum)."
  exit 0
fi

echo ""
echo "Local and server differ. Mismatched files:"
echo "$DIFF_FILES"
echo ""
echo "Server->Local: bash scripts/sync_from_psu.sh"
echo "Local->Server: bash scripts/sync_to_psu.sh"
exit 1
