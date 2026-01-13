#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

# Kaggle 데이터셋 다운로드 및 압축 해제
DATASET="vigneshvenkateswaran/bot-iot-5-data"
OUTPUT_DIR="data/raw"

echo "📥 Downloading Bot-IoT dataset from Kaggle..."
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR"

echo "📦 Extracting dataset..."
cd "$OUTPUT_DIR"
unzip -o -q bot-iot-5-data.zip -d Bot-IoT/ 2>/dev/null || unzip -o -q bot-iot-5-data.zip -d Bot-IoT/
rm -f bot-iot-5-data.zip

echo "✅ Dataset downloaded to $OUTPUT_DIR/Bot-IoT/"
cd ../../