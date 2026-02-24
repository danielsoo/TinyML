#!/bin/bash
# Pull TinyML-main code from PSU server to local.
# Usage: bash scripts/sync_from_psu.sh

set -e
cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== PSU server -> Local (pull code) ==="
echo "Server: $SERVER:$REMOTE"
echo "Local:  $LOCAL_ROOT"
echo ""

rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/raw' --exclude='data/processed' --exclude='* 2.*' --exclude='* 2' --exclude='TinyML-results' --exclude='outputs' "$SERVER:$REMOTE/" "$LOCAL_ROOT/"

echo ""
echo "Server code copied to local."
echo "Verify local vs server: bash scripts/verify_sync.sh"
