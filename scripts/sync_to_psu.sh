#!/bin/bash
# Run from local Mac: sync TinyML-main code to PSU server
# Usage: ./scripts/sync_to_psu.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== TinyML-main -> PSU server sync ==="
echo "Local: $ROOT"
echo "Remote: $SERVER:$REMOTE"
echo ""

rsync -avz \
  --exclude='data/raw' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='data/processed' \
  --exclude='/models' \
  --exclude='outputs' \
  --exclude='TinyML-results' \
  --exclude='*.pyc' \
  ./ "$SERVER:$REMOTE/"

echo ""
echo "Sync complete. On server run:"
echo "  ssh $SERVER"
echo "  conda activate /scratch/yqp5187/conda_envs/research"
echo "  cd $REMOTE"
echo "  python scripts/test_client_fit.py --config config/federated_scratch.yaml  # client test first"
echo "  bash scripts/run_psu_server.sh config/federated_scratch.yaml"
