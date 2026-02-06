#!/bin/bash
# Server -> TinyML-main. Pull all results (analysis, models, eval, outputs) into main.
# One folder per run: data/processed/runs/<version>/<datetime>/analysis/, models/, outputs/, eval/
#
# Usage: bash scripts/pull_results_to_tinyml_results.sh  (from TinyML-main folder)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/.." && pwd)"
PROCESSED_DEST="$DEST/data/processed/runs"
MODELS_DEST="$DEST/models"
OUTPUTS_DEST="$DEST/outputs"
SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== Server -> TinyML-main (pull all results into main) ==="
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
echo "Done. One run = one folder:"
echo "  $DEST/data/processed/runs/<version>/<datetime>/"
echo "    - analysis/, models/, outputs/, eval/"
echo ""
