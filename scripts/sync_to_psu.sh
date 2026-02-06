#!/bin/bash
# Run from local Mac: push TinyML-main code (config, scripts, src) to PSU server.
# Usage: bash scripts/sync_to_psu.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== Local -> PSU server (push code) ==="
echo "Local:  $ROOT"
echo "Remote: $SERVER:$REMOTE"
echo ""

rsync -avz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='data/raw' \
  --exclude='data/processed' \
  --exclude='/models' \
  --exclude='src/models/*.h5' \
  --exclude='src/models/*.tflite' \
  --exclude='outputs' \
  --exclude='TinyML-results' \
  --exclude='processed' \
  --exclude='* 2.*' \
  --exclude='* 2' \
  ./ "$SERVER:$REMOTE/"

echo ""
echo "Sync complete."
echo ""
echo "--- Where to find models on server ---"
echo "  After training (train.py default):"
echo "    $REMOTE/src/models/global_model.h5          (latest copy)"
echo "    $REMOTE/src/models/global_model_YYYYMMDD_HHMMSS.h5  (timestamped file)"
echo "  After SSH to server:"
echo "    ssh $SERVER"
echo "    ls -la $REMOTE/src/models/"
echo "  For 9:1 evaluation only:"
echo "    cd $REMOTE"
echo "    conda activate /scratch/yqp5187/conda_envs/research"
echo "    python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5 --ratio 9"
echo ""
