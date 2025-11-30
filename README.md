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
| **TinyML** | Compress the global model to fit on microcontrollers (≤ few 100 KB) | ⏳ Baseline MLP ready for export; TinyML pipeline in progress |
| **Adversarial Robustness** | Harden the model against evasion/poisoning attacks | ⏳ FGSM utilities scaffolded; integration scheduled for Phase 4 |

**Phase 1 Milestones (Weeks 1–3)**

- ✅ Literature review & requirements analysis  
- ✅ Bot-IoT dataset ingestion + preprocessing pipeline (`load_bot_iot`)  
- ✅ Flower-based FL simulation scaffold  
- ⏳ TinyML "Hello World" deployment test  
- ⏳ README + architecture diagram (this document)  

---

## Project Progress

### Phase 1: Foundation and Setup (Weeks 1-3) - ✅ **Mostly Complete**

**Objective:** Build foundational knowledge and prepare the project environment

| Task | Status | Details |
|------|--------|---------|
| Literature review & requirements analysis | ✅ Complete | Project proposal and literature review completed |
| Bot-IoT dataset ingestion & preprocessing | ✅ Complete | `load_bot_iot()` implemented in `src/data/loader.py`, CSV parsing and preprocessing pipeline completed |
| Flower-based FL simulation scaffold | ✅ Complete | `src/federated/server.py` and `src/federated/client.py` implemented with FedAvg strategy |
| GitHub repository setup | ✅ Complete | Project structure and configuration files set up |
| TinyML "Hello World" deployment | ⏳ In Progress | Basic TFLite export (`src/tinyml/export_tflite.py`) implemented, actual microcontroller deployment testing not completed |

**Completion: ~85%**

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
| Quantization | ⏳ Basic Implementation | Only basic TFLite conversion implemented in `src/tinyml/export_tflite.py`, 8-bit quantization not applied |
| TFLite model export | ⏳ Basic Implementation | Only basic conversion functionality available, optimization options not applied |
| Size vs. Accuracy trade-off analysis | ❌ Not Started | Performance analysis across compression stages not completed |

**Completion: ~20%**

---

### Phase 4: Adversarial Hardening & Deployment (Weeks 12-14) - ⏳ **Preparation Stage**

**Objective:** Integrate adversarial training and deploy the final model

| Task | Status | Details |
|------|--------|---------|
| FGSM attack implementation | ⏳ Basic Utilities Only | Only basic perturbation function implemented in `src/adversarial/fgsm_hook.py` |
| FGSM integration into FL training loop | ❌ Not Started | Adversarial example generation integration into client training loop not completed |
| Adversarial training in FL | ❌ Not Started | Full FL process re-run with adversarial training not completed |
| Re-compression of robust model | ❌ Not Started | Compression pipeline re-run on hardened model not completed |
| Microcontroller deployment | ❌ Not Started | Actual hardware deployment (ESP32/Raspberry Pi Pico) and performance measurement not completed |

**Completion: ~5%**

---

### Phase 5: Final Evaluation & Reporting (Week 15) - ❌ **Not Started**

**Objective:** Quantify project success and deliver final results

| Task | Status | Details |
|------|--------|---------|
| Comprehensive experiments | ❌ Not Started | Comparative experiments between final model and baselines not completed |
| Performance metrics (accuracy, F1-score) | ⏳ Partially Implemented | Metric collection possible in FL simulation, but comprehensive analysis not completed |
| Efficiency metrics (size, latency) | ❌ Not Started | Model size and latency measurement not completed |
| Final project report | ❌ Not Started | Final report writing not started |
| Final presentation & demonstration | ❌ Not Started | Final presentation and demo preparation not started |

**Completion: 0%**

---

### Overall Project Progress Summary

| Phase | Completion | Key Achievements |
|-------|------------|------------------|
| Phase 1: Foundation and Setup | ~85% | Dataset preprocessing and FL basic structure completed |
| Phase 2: Federated Learning Framework | 100% | Complete FL simulation system built |
| Phase 3: TinyML Model Miniaturization | ~20% | Only basic TFLite export implemented |
| Phase 4: Adversarial Hardening & Deployment | ~5% | Only FGSM utilities prepared |
| Phase 5: Final Evaluation & Reporting | 0% | Not started |

**Overall Project Completion: ~42%**

**Next Priorities:**
1. Complete Phase 3: Implement Knowledge Distillation, Pruning, and Quantization
2. Start Phase 4: Integrate FGSM and adversarial training
3. Prepare Phase 5: Design experiments and evaluation framework

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
├── scripts/
│   ├── download_dataset.sh # Kaggle-powered dataset bootstrap
│   └── run_fl_sim.sh       # Convenience wrapper to launch FL simulation
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

> 📖 **Minimal Setup Guide**: See [`docs/MINIMAL_SETUP.md`](docs/MINIMAL_SETUP.md) for the minimum files/folders needed to start training.

1. **Create and populate a virtual environment**
   ```bash
   make setup
   source .venv/bin/activate
   ```

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

Use the separate `requirements_colab.txt` to avoid macOS-specific packages when installing on Linux-based runtimes such as Google Colab.

**Note:** The project now uses separate configuration files for local and Colab environments:
- `config/federated_local.yaml` - For local/macOS execution (uses `data/raw/Bot-IoT`)
- `config/federated_colab.yaml` - For Google Colab execution (uses `/content/drive/MyDrive/TinyML_models`)

> 📖 **Complete Colab Setup Guide**: See [`docs/COLAB_SETUP_GUIDE.md`](docs/COLAB_SETUP_GUIDE.md) for detailed step-by-step instructions from runtime setup to terminal usage.

1. **Clone the repo and install dependencies**
   ```python
   !git clone https://github.com/danielsoo/TinyML.git
   %cd TinyML
   !pip install -r requirements_colab.txt
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

If you prefer using terminal commands directly in Colab:

```python
# 1. Setup
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
%cd /content/TinyML

# 2. Install dependencies
!pip install -r requirements.txt
!pip install flwr[simulation]

# 3. Mount Google Drive (if data is there)
from google.colab import drive
drive.mount('/content/drive')

# 4. Update config
import yaml
with open('config/federated_colab.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['path'] = '/content/drive/MyDrive/TinyML_models'
with open('config/federated_colab.yaml', 'w') as f:
    yaml.dump(cfg, f)

# 5. Run training
!python -m src.federated.client --config config/federated_colab.yaml

# 6. Run analysis
!python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
    --config config/federated_colab.yaml
```
   - The notebook automatically uses `config/federated_colab.yaml` for Colab-specific paths.

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

**Local execution (uses `federated_local.yaml` by default):**
```bash
make run-fl                             # Uses config/federated_local.yaml
# or specify config explicitly
python -m src.federated.client --config config/federated_local.yaml --save-model src/models/global_model.h5
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
- `src/data/loader.py` → `load_bot_iot()` ingests & normalizes Bot-IoT (numeric features only, label = intrusion).
- `partition_non_iid()` scatters data across `num_clients`, creating label-skewed partitions to mimic heterogeneous IoT fleets.
- `src/models/nets.make_mlp()` builds a lightweight MLP tailored for tabular data.
- `src/federated/client.KerasClient` manages Flower’s fit/evaluate cycle and prints detailed metrics each round.

---

## TinyML Export (Baseline)

Once a model is trained centrally (or after FL aggregation), it can be exported to TFLite:

```bash
# Use the saved .h5 from the federated run or generate a fresh one
python -m src.tinyml.export_tflite
```

Output appears in `data/processed/tiny_model.tflite`. A dedicated TinyML "Hello World" deployment script (e.g., Raspberry Pi Pico / ESP32) is scheduled for Phase 1 completion.

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
- Compression ratio (if baseline provided)
- Accuracy, Precision, Recall, F1-Score
- Inference latency (avg/min/max in ms)
- Samples per second

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

## Adversarial Robustness (Preview)

- `src/adversarial/fgsm_hook.py` contains primitive FGSM perturbation utilities.
- Phase 4 will integrate adversarial example generation into the FL training loop and re-run the TinyML compression stage on the hardened global model.

---

## Troubleshooting & Tips

- **MacBook memory pressure?** Lower `max_samples`, or temporarily switch to `placeholder_mnist`.
- **Dataset missing?** Ensure `make download-data` completed, and `data/raw/Bot-IoT/` contains four `reduced_data_*.csv` files.
- **Kaggle CLI “command not found”?** Reactivate the virtual environment (`source .venv/bin/activate`) before running the download script.
- **Long training times?** Prefer lab hardware for full Bot-IoT runs; keep laptop tests to ≤ 10 k samples.

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for the complete text.
