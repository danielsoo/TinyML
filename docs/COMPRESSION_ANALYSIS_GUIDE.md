# Compression Analysis Guide

This guide explains how to use the compression analysis scripts in both **local** and **Google Colab** environments.

## Overview

The compression analysis tools measure:
- **Model size** (file size, parameter count, compression ratio)
- **Accuracy metrics** (Accuracy, Precision, Recall, F1-Score)
- **Inference speed** (latency, samples per second)

---

## 🖥️ Local Environment Usage

### Prerequisites

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Ensure you have a trained model:**
   ```bash
   # Train federated model (if not already done)
   make run-fl
   # This creates: src/models/global_model.h5
   ```

### Basic Usage

#### 1. Analyze Single Model (Baseline)

```bash
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
    --config config/federated_local.yaml \
    --output-dir data/processed/analysis
```

#### 2. Analyze Multiple Models (Baseline + Compressed)

```bash
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "TFLite:data/processed/tiny_model.tflite" \
             "Quantized:data/processed/model_quantized.tflite" \
    --baseline src/models/global_model.h5 \
    --config config/federated_local.yaml \
    --output-dir data/processed/analysis
```

#### 3. Visualize Results

```bash
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --output-dir data/processed/analysis \
    --plot all
```

### Using Makefile (Optional)

```bash
# Analyze (set MODELS variable)
make analyze-compression MODELS="Baseline:src/models/global_model.h5 TFLite:model.tflite"

# Visualize
make visualize-results
```

### Output Files

After running analysis, you'll find:
- `data/processed/analysis/compression_analysis.csv` - Tabular data
- `data/processed/analysis/compression_analysis.json` - JSON format
- `data/processed/analysis/compression_analysis.md` - Markdown report

After visualization:
- `data/processed/analysis/size_vs_accuracy.png` - Size vs accuracy plot
- `data/processed/analysis/compression_metrics.png` - Metrics comparison
- `data/processed/analysis/compression_ratio.png` - Compression ratios

---

## ☁️ Google Colab Usage

### Step-by-Step Guide

#### 1. Setup (Already in train_colab.ipynb)

The Colab notebook already includes:
- Google Drive mounting
- Repository cloning
- Dependency installation

#### 2. Train Model (if needed)

Run the federated learning training cells in `colab/train_colab.ipynb` to generate:
- `src/models/global_model.h5`

#### 3. Export to TFLite (Optional)

Add this cell to convert model to TFLite:

```python
# Export to TFLite
import tensorflow as tf
from src.models.nets import get_model
import yaml

# Load config to get model info
with open("config/federated_colab.yaml") as f:
    cfg = yaml.safe_load(f)

# Load trained model
model = tf.keras.models.load_model("src/models/global_model.h5")

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
tflite_path = "src/models/global_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model: {tflite_path}")
```

#### 4. Run Compression Analysis

Add this cell to analyze models:

```python
# Compression Analysis
!python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "TFLite:src/models/global_model.tflite" \
    --baseline src/models/global_model.h5 \
    --config config/federated_colab.yaml \
    --output-dir data/processed/analysis \
    --format all
```

#### 5. Visualize Results

Add this cell to generate visualizations:

```python
# Visualize Results
!python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --output-dir data/processed/analysis \
    --plot all
```

#### 6. Download Results

Add this cell to download results to Google Drive:

```python
# Copy results to Google Drive
import shutil
from google.colab import drive

drive.mount('/content/drive')

analysis_dir = "data/processed/analysis"
drive_dir = "/content/drive/MyDrive/TinyML_models/analysis"

# Copy all analysis files
import os
os.makedirs(drive_dir, exist_ok=True)

for file in os.listdir(analysis_dir):
    src = os.path.join(analysis_dir, file)
    dst = os.path.join(drive_dir, file)
    if os.path.isfile(src):
        shutil.copy(src, dst)
        print(f"Copied: {file}")

print(f"\n✅ Results saved to: {drive_dir}")
```

#### 7. Display Visualizations in Notebook

Add this cell to display plots inline:

```python
# Display visualizations
from IPython.display import Image, display
import os

analysis_dir = "data/processed/analysis"
plots = [
    "size_vs_accuracy.png",
    "compression_metrics.png",
    "compression_ratio.png"
]

for plot in plots:
    plot_path = os.path.join(analysis_dir, plot)
    if os.path.exists(plot_path):
        print(f"\n## {plot}")
        display(Image(plot_path))
```

---

## 📊 Example Workflow

### Complete Local Workflow

```bash
# 1. Train model
make run-fl

# 2. (Optional) Export to TFLite
python -m src.tinyml.export_tflite

# 3. Analyze models
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "TFLite:data/processed/tiny_model.tflite" \
    --baseline src/models/global_model.h5

# 4. Visualize
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv

# 5. View results
cat data/processed/analysis/compression_analysis.md
open data/processed/analysis/size_vs_accuracy.png
```

### Complete Colab Workflow

1. Open `colab/train_colab.ipynb` in Colab
2. Run all cells up to training
3. Add the analysis cells (steps 3-7 above)
4. Run analysis cells
5. Download results to Drive

---

## 🔧 Advanced Options

### Custom Output Format

```bash
# Only CSV
python scripts/analyze_compression.py \
    --models "Baseline:model.h5" \
    --format csv

# Only JSON
python scripts/analyze_compression.py \
    --models "Baseline:model.h5" \
    --format json
```

### Specific Visualizations

```bash
# Only size vs accuracy plot
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --plot size-accuracy

# Only metrics comparison
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --plot metrics
```

### Custom Output Directory

```bash
python scripts/analyze_compression.py \
    --models "Baseline:model.h5" \
    --output-dir /path/to/custom/output
```

---

## 📝 Notes

### Model Path Format

- **Simple format**: `model.h5` (stage name = filename without extension)
- **Named format**: `Baseline:model.h5` (explicit stage name)

### Config Files

- **Local**: Use `config/federated_local.yaml`
- **Colab**: Use `config/federated_colab.yaml`

### File Paths

- **Local**: Relative paths work (e.g., `src/models/global_model.h5`)
- **Colab**: Use absolute paths or paths relative to project root (e.g., `/content/TinyML/src/models/global_model.h5`)

---

## 🐛 Troubleshooting

### Model Not Found

```
Error: Model file not found: path/to/model.h5
```

**Solution**: Check the path is correct. In Colab, use absolute paths.

### Config File Not Found

```
Error: Configuration file not found
```

**Solution**: Ensure you're using the correct config file:
- Local: `config/federated_local.yaml`
- Colab: `config/federated_colab.yaml`

### Dataset Loading Issues

```
Error: Failed to load dataset
```

**Solution**: 
- Local: Check `data/raw/Bot-IoT/` contains CSV files
- Colab: Ensure Google Drive is mounted and data path is correct

---

## 📚 Additional Resources

- [Matplotlib Documentation](https://matplotlib.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)

