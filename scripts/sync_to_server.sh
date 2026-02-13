#!/bin/bash
# Sync TinyML to server (vast.ai etc). Excludes .git and caches; includes data/raw.
# Usage: SSH_HOST=IP SSH_PORT=PORT REMOTE_PATH=PATH bash scripts/sync_to_server.sh
# Example: SSH_HOST=217.138.104.222 SSH_PORT=10954 REMOTE_PATH=/workspace/TinyML bash scripts/sync_to_server.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_PATH="${REMOTE_PATH:-/workspace/TinyML}"

if [ -z "$SSH_HOST" ]; then
  echo "Usage: SSH_HOST=<ip> SSH_PORT=<port> REMOTE_PATH=<path> $0"
  echo "Example: SSH_HOST=217.138.104.222 SSH_PORT=10954 REMOTE_PATH=/workspace/TinyML $0"
  exit 1
fi

RSYNC_RSH="ssh -p $SSH_PORT"
DEST="root@${SSH_HOST}:${REMOTE_PATH}"

echo "=== TinyML to Server (data/raw included, .git excluded) ==="
echo "Local:  $ROOT"
echo "Remote: $DEST"
echo ""

rsync -avz --progress \
  -e "$RSYNC_RSH" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.DS_Store' \
  --exclude='.cursor' \
  --exclude='.stfolder' \
  --exclude='data/processed/runs' \
  --exclude='TinyML-results' \
  --exclude='outputs' \
  --exclude='processed' \
  --exclude='*.log' \
  ./ "$DEST/"

echo ""
echo "Sync done. data/raw (e.g. CIC-IDS2017) included."
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST -L 8080:localhost:8080"
echo "Then: cd $REMOTE_PATH && python scripts/check_gpu.py"
echo ""
