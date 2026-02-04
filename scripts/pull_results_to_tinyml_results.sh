#!/bin/bash
# Server -> TinyML-main. 결과(분석·모델·eval·outputs) 전부 메인 안으로.
# Run별로 한 폴더: data/processed/runs/<version>/<datetime>/analysis/, models/, outputs/, eval/
#
# Usage: bash scripts/pull_results_to_tinyml_results.sh  (TinyML-main 폴더에서)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/.." && pwd)"
PROCESSED_DEST="$DEST/data/processed/runs"
MODELS_DEST="$DEST/models"
OUTPUTS_DEST="$DEST/outputs"
SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== Server -> TinyML-main (결과 전부 메인 안으로) ==="
echo "From: $SERVER:$REMOTE"
echo "To:   $DEST"
echo ""

mkdir -p "$PROCESSED_DEST"
mkdir -p "$MODELS_DEST"
mkdir -p "$OUTPUTS_DEST"

rsync -avz "$SERVER:$REMOTE/data/processed/runs/" "$PROCESSED_DEST/"
rsync -avz "$SERVER:$REMOTE/models/" "$MODELS_DEST/"
rsync -avz "$SERVER:$REMOTE/outputs/" "$OUTPUTS_DEST/"

echo ""
echo "Done. 한 run = 한 폴더:"
echo "  $DEST/data/processed/runs/<version>/<datetime>/"
echo "    - analysis/, models/, outputs/, eval/"
echo ""
