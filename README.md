# Federated & Adversarially Robust TinyML for IoT Security

This repository implements a full pipeline combining **Federated Learning (FL)**, **TinyML model compression**, and **adversarial attack testing** for privacy-preserving, attack-resilient intrusion detection on resource-constrained IoT hardware.

> **Supervisors**
>
> - Dr. Peilong Li — Associate Professor of Computer Science, *Elizabethtown College*
> - Dr. Suman Saha — Assistant Professor of Computer Science, *Penn State University*

---

## Project Overview

| Component | Description | Status |
|-----------|-------------|--------|
| **Federated Learning** | Train IDS models across distributed IoT clients without sharing raw data | ✅ Flower simulation with FedAvg/FedAvgM, CIC-IDS2017 support |
| **TinyML Compression** | Compress the global model to fit on microcontrollers (pruning, INT8 quantization) | ✅ Full pipeline in `compression.py`; analysis reports in `data/processed/runs/<version>/<run_id>/analysis/` |
| **Adversarial Testing** | FGSM attack evaluation on trained models | ✅ Integrated in `run.py`; reports saved to `run_dir/fgsm/` |

**Primary Dataset:** CIC-IDS2017 (binary BENIGN vs ATTACK). Bot-IoT supported via data loader.

---

## Quick Start

```bash
# Full pipeline: train → compress → analyze → FGSM → ratio sweep → visualize
python run.py --config config/federated.yaml

# Skip training (use existing model)
python run.py --config config/federated.yaml --skip-train

# Skip FGSM attack test
python run.py --config config/federated.yaml --skip-fgsm
```

---

## Pipeline (`run.py`)

The main runner executes:

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/train.py` | Federated training → `src/models/global_model.h5` |
| 2 | `compression.py` | Pruning, quantization → `models/tflite/saved_model_*.tflite` |
| 3 | `scripts/analyze_compression.py` | Compression analysis → `run_dir/analysis/compression_analysis.md` |
| 3b | `scripts/test_fgsm_attack.py` | FGSM attack test → `run_dir/fgsm/fgsm_report.md`, `fgsm_results.json` |
| 4 | `scripts/evaluate_ratio_sweep.py` | Normal:attack ratio sweep |
| 4b | `scripts/tune_threshold_all_ratios.py` | Threshold tuning (appended to ratio sweep report) |
| 5 | `scripts/visualize_results.py` | Plots (size vs accuracy, compression metrics, etc.) |

**Run output structure:** `data/processed/runs/<version>/<run_id>/`

```
runs/<version>/<run_id>/
├── run_config.yaml       # Run configuration snapshot
├── models/               # Keras + TFLite models (copied after pipeline)
├── analysis/             # Compression analysis (CSV, JSON, MD, PNG)
├── fgsm/                 # FGSM attack report and results
└── eval/                 # Ratio sweep report
```

---

## Repository Layout

```
TinyML/
├── run.py                    # Main pipeline runner
├── compression.py            # TinyML compression pipeline (prune → quantize → TFLite)
├── config/
│   ├── federated.yaml        # FL training config (CIC-IDS2017, focal loss, FedAvgM)
│   ├── federated_local.yaml  # Local/macOS config (Bot-IoT)
│   └── fgsm.yaml             # FGSM attack test config
├── data/
│   ├── raw/                  # Raw datasets (CIC-IDS2017, Bot-IoT)
│   └── processed/
│       └── runs/             # Run outputs: <version>/<run_id>/{analysis,fgsm,eval,models}
├── docs/                     # Setup guides, task docs
├── esp32_tflite_project/     # ESP32 PlatformIO project for TFLite Micro
├── scripts/
│   ├── train.py              # FL training entry point
│   ├── analyze_compression.py # Compression stage analysis and reports
│   ├── test_fgsm_attack.py   # FGSM adversarial attack testing
│   ├── evaluate_ratio_sweep.py # Normal:attack ratio evaluation
│   ├── tune_threshold_all_ratios.py # Threshold tuning per ratio
│   ├── visualize_results.py  # Size vs accuracy, metrics plots
│   ├── create_test_model.py  # Test model for microcontroller
│   ├── deploy_microcontroller.py # TFLite to C array conversion
│   └── test_tflite_inference.py # Local TFLite inference test
├── src/
│   ├── adversarial/          # FGSM hooks (fgsm_hook.py)
│   ├── data/                 # Dataset loaders (CIC-IDS2017, Bot-IoT)
│   ├── federated/            # Flower client/server logic
│   ├── modelcompression/     # Distillation, pruning, quantization
│   ├── models/               # MLP, CNN model definitions
│   └── tinyml/               # TFLite export
├── tests/                    # Unit tests
└── README.md
```

---

## Configuration

### `config/federated.yaml`

- **Data:** CIC-IDS2017, binary (BENIGN vs ATTACK), SMOTE, balance_ratio 4.0
- **Evaluation:** `prediction_threshold` (default 0.3 for binary classification)
- **Federated:** FedAvgM, 80 rounds, focal loss, QAT, learning rate decay (cosine)

### `config/fgsm.yaml`

- **Data:** Same as federated (CIC-IDS2017, max_samples 2M)
- **Attack:** Epsilon sweep [0.01, 0.05, 0.1, 0.15, 0.2], `prediction_threshold` 0.3
- **Eval:** test_subset_size 5000, adv_subset_size 20000

When the model path is under `data/processed/runs/<version>/<run_id>/models/`, FGSM and compression analysis use that run's `run_config.yaml` for `prediction_threshold` (ensuring consistency with training).

---

## Server (Vast.ai)

**서버에는 TinyML 폴더만 있으면 됩니다.** 워크스페이스 경로: `/workspace/TinyML`

- Git 연결 없이 로컬 **TinyML** 폴더를 서버로 넘기면 됩니다 (Syncthing, scp, zip 등).
- [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md) · [docs/SERVER_SYNC_COMMANDS.md](docs/SERVER_SYNC_COMMANDS.md)

---

## Environment Setup

1. **Create virtual environment**
   ```bash
   make setup
   source .venv/bin/activate   # Linux/macOS
   ```

2. **Download dataset**
   - **CIC-IDS2017:** Place `.pcap_ISCX.csv` files in `data/raw/CIC-IDS2017/`
   - **Bot-IoT:** Use `make download-data` (Kaggle) or place data in `data/raw/Bot-IoT/`

3. **Run pipeline**
   ```bash
   python run.py --config config/federated.yaml
   ```

---

## Federated Learning

```bash
# Train (FL)
python scripts/train.py --config config/federated.yaml

# Centralized training (no FL)
python run.py --config config/federated.yaml --centralized
```

- **Client:** `src/federated/client.py` (KerasClient, focal loss, class weights)
- **Server:** `src/federated/server.py` (FedAvg/FedAvgM, momentum, LR decay)

---

## Compression Pipeline

`compression.py` performs:

1. Load trained Keras model
2. Structured pruning
3. Fine-tuning
4. INT8 quantization (PTQ)
5. TFLite export (float32 + INT8 variants)

Output: `models/tflite/saved_model_original.tflite`, `saved_model_pruned_quantized.tflite`

---

## FGSM Attack Test

```bash
# Run FGSM on a run's model (uses run_config threshold)
python scripts/test_fgsm_attack.py --model data/processed/runs/v17/2026-02-09_17-52-41/models/global_model.h5

# Or let run.py execute it (Step 3b)
python run.py --config config/federated.yaml
```

Reports: `run_dir/fgsm/fgsm_report.md`, `fgsm_results.json`

---

## Microcontroller Deployment

```bash
python scripts/create_test_model.py
python scripts/deploy_microcontroller.py
python scripts/test_tflite_inference.py
```

ESP32 project: `esp32_tflite_project/` (PlatformIO)

> 📖 See [docs/MICROCONTROLLER_DEPLOYMENT.md](docs/MICROCONTROLLER_DEPLOYMENT.md)

---

## Troubleshooting

- **Model load error (TypeError: string indices must be integers):** Model was saved with custom loss (e.g. focal). Use `compile=False` when loading and recompile for evaluation.
- **Keras Precision/Recall/F1 = 0:** Ensure `prediction_threshold` in config/run_config matches training (e.g. 0.3). Compression analysis and FGSM read `run_config.yaml` when the model is under a run dir.
- **Dataset missing:** Check `data/raw/CIC-IDS2017/` or `data/raw/Bot-IoT/` per config.
- **Memory:** Lower `max_samples` in config for large datasets.

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.
