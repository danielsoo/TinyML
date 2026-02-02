#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

# Generate timestamped model filename to preserve training history
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_DIR="src/models"
TIMESTAMPED_MODEL="${MODEL_DIR}/global_model_${TIMESTAMP}.h5"
LATEST_MODEL="${MODEL_DIR}/global_model.h5"

# Create models directory if it doesn't exist
mkdir -p "${MODEL_DIR}"

# Use local config, save with timestamped filename
echo "🚀 Starting federated learning..."
echo "📁 Model will be saved as: ${TIMESTAMPED_MODEL}"
python -m src.federated.client \
    --config config/federated_local.yaml \
    --save-model "${TIMESTAMPED_MODEL}"

# Also save as latest for easy access
if [ -f "${TIMESTAMPED_MODEL}" ]; then
    cp "${TIMESTAMPED_MODEL}" "${LATEST_MODEL}"
    echo ""
    echo "✅ Training complete!"
    echo "   📦 Timestamped model: ${TIMESTAMPED_MODEL}"
    echo "   📌 Latest model: ${LATEST_MODEL}"
else
    echo "⚠️ Model file not found: ${TIMESTAMPED_MODEL}"
fi
