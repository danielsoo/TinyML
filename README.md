## Federated & Adversarially Robust TinyML for IoT Security

This repository hosts the capstone project exploring how **Federated Learning (FL)**, **TinyML model compression**, and **Adversarial Training** can be combined to deliver privacy-preserving and attack-resilient intrusion detection on extremely resource-constrained IoT hardware.

> **Supervisors**
>
> - Dr. Peilong Li — Associate Professor of Computer Science, *Elizabethtown College*  
> - Dr. Suman Saha — Assistant Professor of Computer Science, *Penn State University*

---

## Project Overview

| Pillar | Goal | Current Status |
|--------|------|----------------|
| **Federated Learning** | Train IDS models across distributed IoT clients without sharing raw data | ✅ Flower simulation skeleton implemented |
| **TinyML** | Compress the global model to fit on microcontrollers (≤ few 100 KB) | ✅ Basic TFLite export implemented; compression analysis tools completed |
| **Adversarial Robustness** | Harden the model against evasion/poisoning attacks | ⏳ FGSM utilities scaffolded; integration scheduled for Phase 4 |

**Phase 1 Milestones (Weeks 1–3)**

- ✅ Literature review & requirements analysis  
- ✅ Bot-IoT dataset ingestion + preprocessing pipeline (`load_bot_iot`)  
- ✅ Enhanced preprocessing: IP addresses and categorical features encoding
- ✅ Flower-based FL simulation scaffold  
- ✅ Basic TFLite export functionality
- ✅ README + architecture diagram (this document)  

---

## Project Progress

### Phase 1: Foundation and Setup (Weeks 1-3) - ✅ **Mostly Complete**

**Objective:** Build foundational knowledge and prepare the project environment

| Task | Status | Details |
|------|--------|---------|
| Literature review & requirements analysis | ✅ Complete | Project proposal and literature review completed |
| Bot-IoT dataset ingestion & preprocessing | ✅ Complete | `load_bot_iot()` implemented in `src/data/loader.py` with IP address conversion and categorical feature encoding (45 features), CSV parsing and preprocessing pipeline completed |
| Flower-based FL simulation scaffold | ✅ Complete | `src/federated/server.py` and `src/federated/client.py` implemented with FedAvg strategy |
| GitHub repository setup | ✅ Complete | Project structure and configuration files set up |
| TinyML basic export | ✅ Complete | Basic TFLite export (`src/tinyml/export_tflite.py`) implemented, compression analysis tools available |
| Microcontroller setup & toolchain validation | ✅ Complete | Test model creation (`scripts/create_test_model.py`), TFLite to C array conversion (`scripts/deploy_microcontroller.py`), local inference testing (`scripts/test_tflite_inference.py`), and ESP32 project structure ready. Deployment pipeline validated without hardware |

**Completion: 100%**

---

### Phase 2: Federated Learning Framework (Weeks 4-7) - ✅ **Complete**

**Objective:** Develop and simulate the privacy-preserving training framework

| Task | Status | Details |
|------|--------|---------|
| Central server with FedAvg algorithm | ✅ Complete | FedAvg strategy implemented in `src/federated/server.py` with configuration-based round management |
| FL client-side training logic | ✅ Complete | Local training/evaluation implemented via `KerasClient` class in `src/federated/client.py` |
| Data loaders & partitioning | ✅ Complete | Non-IID partitioning (`partition_non_iid`) implemented in `src/data/loader.py` |
| FL simulation with virtual clients | ✅ Complete | Multi-client training and global model generation possible via Flower simulation |
| Model export & saving | ✅ Complete | Trained global model saving functionality implemented in `.h5` format |
| Evaluation metrics | ✅ Complete | Detailed metrics output including Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

**Completion: 100%**

---

### Phase 3: TinyML Model Miniaturization (Weeks 8-11) - ⏳ **Partially Complete**

**Objective:** Compress the global model to fit on microcontrollers

| Task | Status | Details |
|------|--------|---------|
| Knowledge Distillation | ❌ Not Started | Teacher-Student model architecture and training logic not implemented |
| Structured Pruning | ❌ Not Started | Model reduction functionality via filter/neuron removal not implemented |
| Quantization | ⏳ Basic Implementation | Basic TFLite conversion implemented, 8-bit quantization (INT8) not yet applied |
| TFLite model export | ✅ Complete | Basic TFLite conversion implemented in `src/tinyml/export_tflite.py`, supports H5 → TFLite conversion |
| Size vs. Accuracy trade-off analysis | ✅ Complete | Comprehensive analysis tools implemented: `scripts/analyze_compression.py` and `scripts/visualize_results.py` with CSV/JSON/Markdown reports and visualization plots |

**Completion: ~50%**

---

### Phase 4: Adversarial Hardening & Deployment (Weeks 12-14) - ⏳ **Preparation Stage**

**Objective:** Integrate adversarial training and deploy the final model

| Task | Status | Details |
|------|--------|---------|
| FGSM attack implementation | ⏳ Basic Utilities Only | Only basic perturbation function implemented in `src/adversarial/fgsm_hook.py` |
| FGSM integration into FL training loop | ❌ Not Started | Adversarial example generation integration into client training loop not completed |
| Adversarial training in FL | ❌ Not Started | Full FL process re-run with adversarial training not completed |
| Re-compression of robust model | ❌ Not Started | Compression pipeline re-run on hardened model not completed |
| Microcontroller deployment | ⏳ In Progress | Test model creation, C array conversion, and local validation complete. ESP32 project structure ready. Hardware deployment pending hardware availability |

**Completion: ~5%**

---

### Phase 5: Final Evaluation & Reporting (Week 15) - ❌ **Not Started**

**Objective:** Quantify project success and deliver final results

| Task | Status | Details |
|------|--------|---------|
| Comprehensive experiments | ⏳ In Progress | Compression analysis framework implemented, ready for multi-stage comparisons |
| Performance metrics (accuracy, F1-score) | ✅ Complete | Metric collection implemented in FL simulation and compression analysis (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) |
| Efficiency metrics (size, latency) | ✅ Complete | Model size (MB, bytes, parameters), inference latency (avg/min/max), and compression ratios measurement implemented in `scripts/analyze_compression.py` |
| Analysis reports & visualizations | ✅ Complete | Comprehensive reports in CSV/JSON/Markdown formats with baseline comparisons, visualization plots (size vs accuracy, metrics comparison, compression ratios) |
| Final project report | ❌ Not Started | Final report writing not started |
| Final presentation & demonstration | ❌ Not Started | Final presentation and demo preparation not started |

**Completion: ~40%**

---

### Overall Project Progress Summary

| Phase | Completion | Key Achievements |
|-------|------------|------------------|
| Phase 1: Foundation and Setup | 100% | Enhanced dataset preprocessing (IP addresses, categorical features), FL simulation, basic TFLite export, and microcontroller toolchain validation completed |
| Phase 2: Federated Learning Framework | 100% | Complete FL simulation system built with detailed metrics |
| Phase 3: TinyML Model Miniaturization | ~50% | Basic TFLite export and comprehensive compression analysis tools completed |
| Phase 4: Adversarial Hardening & Deployment | ~15% | FGSM utilities prepared. Microcontroller deployment pipeline validated (test model, C conversion, ESP32 project ready) |
| Phase 5: Final Evaluation & Reporting | ~40% | Performance and efficiency metrics collection, analysis reports, and visualizations completed |

**Overall Project Completion: ~58%**

**Next Priorities:**
1. Complete Phase 3: Implement Knowledge Distillation, Structured Pruning, and 8-bit Quantization (INT8)
2. Start Phase 4: Integrate FGSM into FL training loop and implement adversarial training
3. Prepare Phase 5: Conduct comprehensive experiments and finalize reporting

---

## Repository Layout

```
TinyML/
├── config/                 # YAML configs (e.g., FL hyperparameters)
│   ├── federated_local.yaml  # Local/macOS configuration
│   └── federated_colab.yaml  # Google Colab configuration
├── data/
│   ├── raw/                # Raw datasets (Bot-IoT ZIP extracts go here)
│   └── processed/          # Exported TFLite / intermediate artifacts
│       └── microcontroller/ # Microcontroller deployment files
├── esp32_tflite_project/   # ESP32 PlatformIO project
│   ├── platformio.ini      # PlatformIO configuration
│   └── src/                # ESP32 source code
├── scripts/
│   ├── download_dataset.sh # Kaggle-powered dataset bootstrap
│   ├── run_fl_sim.sh       # Convenience wrapper to launch FL simulation
│   ├── create_test_model.py # Create test model for microcontroller
│   ├── deploy_microcontroller.py # Convert TFLite to C array
│   └── test_tflite_inference.py # Test inference locally
├── src/
│   ├── adversarial/        # FGSM hooks and upcoming defenses
│   ├── data/               # Dataset loaders, partitioning utilities
│   ├── federated/          # Flower client/server logic
│   ├── models/             # Model definitions (MLP, CNN baseline, etc.)
│   └── tinyml/             # TFLite export tooling
└── README.md
```

---

## Environment Setup

> macOS 15 with Apple Silicon is the reference development environment.
> The project now supports both local and Colab environments with automatic detection.

> 📖 **Minimal Setup Guide**: See [`docs/MINIMAL_SETUP.md`](docs/MINIMAL_SETUP.md) for the minimum files/folders needed to start training.

1. **Create and populate a virtual environment** (Local only)
   ```bash
   make setup
   source .venv/bin/activate
   ```
   
   **Note:** In Colab, dependencies are automatically installed by the unified training script.

2. **Authenticate with Kaggle (one-time)**
   ```bash
   pip install kaggle  # already in requirements.txt
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the Bot-IoT dataset (5 % Kaggle subset by default)**
   ```bash
   make download-data
   ```
   - Archives land in `data/raw/Bot-IoT/`
   - To re-download, rerun the same command (it overwrites safely)

---

## Running on Google Colab

The project automatically detects the environment (local vs Colab) and uses the appropriate configuration.

**Note:** The project uses separate configuration files for local and Colab environments:
- `config/federated_local.yaml` - For local/macOS execution (uses `data/raw/Bot-IoT`)
- `config/federated_colab.yaml` - For Google Colab execution (uses `/content/drive/MyDrive/TinyML_models`)

**Environment Auto-Detection:**
- The unified training script (`scripts/train.py`) automatically detects whether you're running locally or in Colab
- Configuration files are selected automatically based on the environment
- No need to manually specify config files in most cases

> 📖 **Complete Colab Setup Guide**: See [`docs/COLAB_SETUP_GUIDE.md`](docs/COLAB_SETUP_GUIDE.md) for detailed step-by-step instructions from runtime setup to terminal usage.

1. **Clone the repo and run training (unified script)**
   ```python
   !git clone https://github.com/danielsoo/TinyML.git
   %cd TinyML
   !python scripts/train.py
   ```
   
   The unified script automatically:
   - Detects Colab environment
   - Installs dependencies (including protobuf fix)
   - Uses `config/federated_colab.yaml`
   - Checks GPU availability
   
   **Or use traditional approach:**
   ```python
   !git clone https://github.com/danielsoo/TinyML.git
   %cd TinyML
   !pip install -r colab/requirements_colab.txt
   !python -m src.federated.client --config config/federated_colab.yaml --save-model src/models/global_model.h5
   ```

2. **Access the dataset**
   - Option A: Upload the Kaggle ZIP to Google Drive and mount it
     ```python
     from google.colab import drive
     drive.mount("/content/drive")
     ```
     Then point `load_bot_iot(data_path="/content/drive/MyDrive/…")` to the mounted directory.
     
   - Option B: Download directly inside Colab using the Kaggle CLI (requires uploading `kaggle.json` just like on macOS).

3. **Run the notebook or simulation**
   ```python
   !python -m src.federated.client --config config/federated_colab.yaml
   ```
   The Colab config file (`federated_colab.yaml`) is pre-configured for Google Drive paths. Adjust `max_samples` in the config for the available GPU/CPU quota. Exported models (e.g., `.h5`, `.tflite`) can be saved to Drive or downloaded via `google.colab.files.download`.

4. **End-to-end Colab workflow**
   - Open `colab/train_colab.ipynb` in Colab to walk through GPU checks, repo sync, dependency installation, dataset prep, training, and Drive backup of exported models.
   - **Quick Start**: Runtime → Change runtime type → GPU, then Runtime → Run all

### Quick Colab Setup (Terminal Commands)

**Recommended: Use unified training script (auto-detects Colab environment)**
```python
# 1. Clone and setup
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
%cd /content/TinyML

# 2. Mount Google Drive (if data is there)
from google.colab import drive
drive.mount('/content/drive')

# 3. Run training (auto-detects Colab, installs deps, uses correct config)
!python scripts/train.py
```

**Alternative: Manual setup**
```python
# 1. Setup
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
%cd /content/TinyML

# 2. Install dependencies
!pip install -r colab/requirements_colab.txt
!pip install flwr[simulation]

# 3. Mount Google Drive (if data is there)
from google.colab import drive
drive.mount('/content/drive')

# 4. Run training
!python -m src.federated.client --config config/federated_colab.yaml --save-model src/models/global_model.h5

# 5. Run analysis
!python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
    --config config/federated_colab.yaml
```
   - The unified script automatically uses `config/federated_colab.yaml` for Colab-specific paths.

---

## Federated Learning Simulation

The Flower simulation spins up multiple virtual IoT clients, each training on a partition of the Bot-IoT dataset. Results include detailed metrics (accuracy, precision/recall, confusion matrix) per round.

### 1. Configure the run

**For Local Execution:**
`config/federated_local.yaml` (default for local runs)
```yaml
data:
  name: "bot_iot"
  path: "data/raw/Bot-IoT"  # Local path
  num_clients: 4
  max_samples: 200000

federated:
  num_rounds: 3
  local_epochs: 2
  batch_size: 128
```

**For Google Colab:**
`config/federated_colab.yaml` (used in Colab notebook)
```yaml
data:
  name: "bot_iot"
  path: "/content/drive/MyDrive/TinyML_models"  # Colab Google Drive path
  num_clients: 4
  max_samples: 200000

federated:
  num_rounds: 3
  local_epochs: 2
  batch_size: 128
```

- `max_samples` caps the number of rows loaded from Bot-IoT to prevent OOM on laptops. Increase gradually (e.g., 50000, then full dataset) once running on lab hardware.
- Switch back to `"mnist"` if you need to smoke-test without the large dataset.

### 2. Launch the simulation

**Local execution (auto-detects environment and uses appropriate config):**
```bash
# Option 1: Use unified training script (recommended)
make train                              # Auto-detects environment, uses appropriate config
# or
python scripts/train.py                 # Same as above

# Option 2: Use shell script (legacy, local only)
make run-fl                             # Uses config/federated_local.yaml
# Automatically saves with timestamp: global_model_YYYYMMDD_HHMMSS.h5
# Also saves as latest: global_model.h5

# Option 3: Direct Python execution
python -m src.federated.client \
    --config config/federated_local.yaml \
    --save-model src/models/my_model.h5

# The unified script automatically:
# - Detects local vs Colab environment
# - Selects appropriate config file (federated_local.yaml or federated_colab.yaml)
# - Checks GPU availability
# - Verifies data paths
# - Handles dependency installation (in Colab)
```

**Colab execution:**
```bash
python -m src.federated.client --config config/federated_colab.yaml --save-model src/models/global_model.h5
```

**During execution you will see logs similar to:**
```
📂 Loading Bot-IoT data from 4 files...
  Loaded reduced_data_1.csv: 1000000 samples
  ...
============================================================
📊 Evaluation Summary
============================================================
Accuracy: 0.9450 (94.50%)
Loss: 0.1421

📈 Ground Truth:
  - Attack samples: 150
  - Normal samples: 50
  - Total samples: 200

🔮 Predictions:
  - Predicted Attack: 147
  - Predicted Normal: 53

✅ Confusion Matrix:
  - True Positives (TP): 144
  - True Negatives (TN): 48
  - False Positives (FP): 5
  - False Negatives (FN): 6

📏 Metrics:
  - Precision: 0.9664 (96.64%)
  - Recall: 0.9600 (96.00%)
  - F1-Score: 0.9632 (96.32%)
============================================================
```

### 3. What happens under the hood?
- `src/data/loader.py` → `load_bot_iot()` ingests & preprocesses Bot-IoT dataset:
  - Converts IP addresses (src/dst IPs) to integers
  - Encodes categorical features (`proto`, `flgs`, `state`, `service`) using Label Encoding
  - Normalizes numeric features (45 total features after encoding)
  - Labels: binary classification (intrusion vs normal)
- `partition_non_iid()` scatters data across `num_clients`, creating label-skewed partitions to mimic heterogeneous IoT fleets.
- `src/models/nets.make_mlp()` builds a lightweight MLP tailored for tabular data (45 input features).
- `src/federated/client.KerasClient` manages Flower's fit/evaluate cycle and prints detailed metrics each round.

---

## TinyML Export (Baseline)

Once a model is trained centrally (or after FL aggregation), it can be exported to TFLite:

```bash
# Use the saved .h5 from the federated run or generate a fresh one
python -m src.tinyml.export_tflite
```

Output appears in `data/processed/tiny_model.tflite`. For comprehensive analysis of compression stages (size, accuracy, latency), see the **Compression Analysis** section below.

---

## Compression Analysis

The compression analysis tools measure model size, accuracy, and inference speed at each compression stage, and visualize size vs accuracy trade-offs.

### 1. Analyze Compression Stages

Analyze multiple model files (baseline, quantized, pruned, etc.) and generate comprehensive reports:

```bash
# Analyze multiple models with stage names
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "Quantized:data/processed/model_quantized.tflite" \
             "Pruned:data/processed/model_pruned.tflite" \
    --baseline src/models/global_model.h5 \
    --config config/federated_local.yaml \
    --output-dir data/processed/analysis

# Or use Makefile (set MODELS variable)
make analyze-compression MODELS="Baseline:src/models/global_model.h5 Quantized:model.tflite"
```

**Output:**
- `compression_analysis.csv` - Tabular results
- `compression_analysis.json` - JSON format for programmatic access
- `compression_analysis.md` - Markdown report with comparison tables

**Metrics Collected:**
- Model file size (MB, bytes)
- Parameter count
- Compression ratio vs baseline (with size reduction percentage)
- Accuracy, Precision, Recall, F1-Score
- Inference latency (avg/min/max in ms)
- Samples per second

**Report Features:**
- Quantitative comparison with baseline (percentage changes for size, accuracy, F1-score, latency)
- Visual indicators (↑ improvement, ↓ degradation, → no change)
- Overall status summary for each compression stage
- Detailed markdown report with comparison tables

### 2. Visualize Results

Generate visualizations from analysis results:

```bash
# Generate all visualizations
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --output-dir data/processed/analysis

# Or generate specific plots
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --plot size-accuracy  # or: metrics, compression-ratio, all

# Or use Makefile
make visualize-results
```

**Generated Plots:**
- `size_vs_accuracy.png` - Size vs accuracy trade-off with trend line
- `compression_metrics.png` - Comprehensive metrics comparison (4 subplots)
- `compression_ratio.png` - Compression ratio visualization

### 3. Local Environment Usage

**Basic workflow:**
```bash
# 1. Train federated model (if not already done)
make run-fl

# 2. Analyze baseline model
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
    --config config/federated_local.yaml

# 3. (Optional) Export to TFLite and analyze
python -m src.tinyml.export_tflite
python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "TFLite:data/processed/tiny_model.tflite" \
    --baseline src/models/global_model.h5

# 4. Visualize results
python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv
```

### 4. Google Colab Usage

**Option A: Use the notebook (Recommended)**

The `colab/train_colab.ipynb` notebook now includes compression analysis cells (Section 8️⃣). Simply run all cells sequentially.

**Option B: Manual execution in Colab**

```python
# In Colab notebook cells:

# 1. Export to TFLite (optional)
import tensorflow as tf
model = tf.keras.models.load_model("src/models/global_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("src/models/global_model.tflite", "wb") as f:
    f.write(tflite_model)

# 2. Run analysis
!python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
             "TFLite:src/models/global_model.tflite" \
    --baseline src/models/global_model.h5 \
    --config config/federated_colab.yaml \
    --output-dir data/processed/analysis

# 3. Visualize
!python scripts/visualize_results.py \
    --results data/processed/analysis/compression_analysis.csv \
    --output-dir data/processed/analysis

# 4. Display plots inline
from IPython.display import Image, display
display(Image("data/processed/analysis/size_vs_accuracy.png"))
```

**For detailed Colab instructions, see:** `docs/COMPRESSION_ANALYSIS_GUIDE.md`

---

## Microcontroller Deployment

The project includes tools for deploying TFLite models to ESP32 microcontrollers. Even without physical hardware, you can validate the deployment pipeline locally.

> 📖 **Complete Deployment Guide**: See [`docs/MICROCONTROLLER_DEPLOYMENT.md`](docs/MICROCONTROLLER_DEPLOYMENT.md) for detailed step-by-step instructions.

### 1. Create Test Model

Generate a simple "Hello World" ML model for microcontroller testing:

```bash
# Activate virtual environment
source .venv/bin/activate

# Create test model
python scripts/create_test_model.py
```

This creates:
- `data/processed/microcontroller/hello_world_model.h5` - Keras model
- `data/processed/microcontroller/hello_world_model.tflite` - TFLite model (2.12 KB)

### 2. Convert to C Array

Convert TFLite model to C array format for ESP32:

```bash
python scripts/deploy_microcontroller.py
```

This generates:
- `data/processed/microcontroller/model_data.c` - C source file
- `data/processed/microcontroller/model_data.h` - C header file

### 3. Test Inference Locally (Without Hardware)

Validate the deployment pipeline without physical hardware:

```bash
# Test TFLite inference
python scripts/test_tflite_inference.py

# Also verify C array files
python scripts/test_tflite_inference.py --verify-c-files
```

This script:
- ✅ Loads and validates the TFLite model
- ✅ Runs inference with multiple test cases
- ✅ Measures inference time
- ✅ Verifies C array file format
- ✅ Confirms deployment readiness

### 4. ESP32 Project Structure

The ESP32 project is ready in `esp32_tflite_project/`:

```
esp32_tflite_project/
├── platformio.ini          # PlatformIO configuration
├── src/
│   ├── main.cpp            # ESP32 main code with TensorFlow Lite Micro
│   ├── model_data.c        # Model C array (auto-generated)
│   └── model_data.h        # Model header (auto-generated)
└── lib/                    # Library directory
```

### 5. Deploy to ESP32 (When Hardware Available)

```bash
cd esp32_tflite_project

# Install libraries (first time only)
pio lib install

# Build
pio run

# Upload to ESP32
pio run --target upload

# Monitor serial output
pio device monitor
```

**Expected Serial Output:**
```
========================================
ESP32 TensorFlow Lite Micro Test
========================================

Loading TFLite model...
Model loaded successfully!
Model size: 2168 bytes

Model initialized successfully!
Input shape: [1, 2]
Output shape: [1, 1]

Ready for inference!
========================================

Running inference...
Input: [0.50, 0.50]
Output: 0.472900
Inference time: 1234 microseconds
```

### Deployment Status

- ✅ Test model creation script
- ✅ TFLite to C array conversion
- ✅ Local inference testing (no hardware required)
- ✅ ESP32 project structure and code
- ⏳ Actual hardware deployment (pending hardware availability)

---

## Adversarial Robustness (Preview)

- `src/adversarial/fgsm_hook.py` contains primitive FGSM perturbation utilities.
- Phase 4 will integrate adversarial example generation into the FL training loop and re-run the TinyML compression stage on the hardened global model.

---

## Troubleshooting & Tips

- **MacBook memory pressure?** Lower `max_samples`, or temporarily switch to `placeholder_mnist`.
- **Dataset missing?** Ensure `make download-data` completed, and `data/raw/Bot-IoT/` contains four `reduced_data_*.csv` files.
- **Kaggle CLI “command not found”?** Reactivate the virtual environment (`source .venv/bin/activate`) before running the download script.
- **Long training times?** Prefer lab hardware for full Bot-IoT runs; keep laptop tests to ≤ 10 k samples.
- **Protobuf version errors in Colab?** The notebook automatically installs `protobuf==3.20.3` for TensorFlow compatibility.
- **Model input shape mismatch?** Ensure your model was trained with the current dataset (45 features including IP addresses and categorical encodings). Retrain if needed.
- **TFLite evaluation errors?** TFLite models use batch size 1 by default; the compression analysis script handles this automatically.

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for the complete text.
