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

## Repository Layout

```
TinyML/
├── config/                 # YAML configs (e.g., FL hyperparameters)
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

## Federated Learning Simulation

The Flower simulation spins up multiple virtual IoT clients, each training on a partition of the Bot-IoT dataset. Results include detailed metrics (accuracy, precision/recall, confusion matrix) per round.

### 1. Configure the run

`config/federated.yaml`
```yaml
server:
  rounds: 3
  min_available_clients: 2

client:
  local_epochs: 1
  batch_size: 64

data:
  name: "bot_iot"
  train_split: 0.8
  num_clients: 4
  max_samples: 10000  # Reduce if running on limited hardware
```

- `max_samples` caps the number of rows loaded from Bot-IoT to prevent OOM on laptops. Increase gradually (e.g., 50000, then full dataset) once running on lab hardware.
- Switch back to `"placeholder_mnist"` if you need to smoke-test without the large dataset.

### 2. Launch the simulation

```bash
make run-fl
# or
python -m src.federated.client
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
python -m src.tinyml.export_tflite
```

Output appears in `data/processed/tiny_model.tflite`. A dedicated TinyML “Hello World” deployment script (e.g., Raspberry Pi Pico / ESP32) is scheduled for Phase 1 completion.

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
