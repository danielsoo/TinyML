# Federated & Adversarially Robust TinyML for IoT Security

End-to-end research code for **privacy-preserving intrusion detection** on resource-constrained devices: **Federated Learning (FL)**, **TinyML compression** (distillation, pruning, QAT/PTQ, TFLite), and **adversarial robustness evaluation** (PGD and optional FGSM utilities).

**Authors:** Younsoo Park, Seokhyeon Bae

> **Supervisors**
>
> - Dr. Peilong Li — Associate Professor of Computer Science, *Elizabethtown College*
> - Dr. Suman Saha — Assistant Professor of Computer Science, *Penn State University*

---

## Highlights

| Area | What this repo does |
|------|---------------------|
| **Federated learning** | Flower-based simulation (FedAvg / FedAvgM), focal loss, optional QAT, cosine LR decay on CIC-IDS2017 (binary) and Bot-IoT via loaders |
| **TinyML compression** | Pruning, knowledge distillation, QAT / PTQ, TFLite export; batch-norm folding and analysis reports |
| **Robustness** | **PGD** attack evaluation on Keras + TFLite models (`run_pgd.py`), shared adversarial examples, optional ε-sweep; legacy **FGSM** script still available |
| **Compression grid sweep** | Optional **48-way** sweep (FL QAT × distillation × pruning × PTQ) + PGD merge + heatmaps (`run_sweep_and_pgd.py`) |
| **Edge deployment** | ESP32 TFLite Micro project under `esp32_tflite_project/` |

**Primary dataset:** CIC-IDS2017 (binary: BENIGN vs ATTACK). Bot-IoT is supported through the data loader.

---

## Quick start

```bash
# Environment (see below for details)
make setup
source .venv/bin/activate   # Linux / macOS

# Full pipeline: FL → compress → analyze → PGD → ratio sweep → threshold tuning → visualize
python run.py --config config/federated.yaml

# Common variants
python run.py --config config/federated.yaml --skip-train          # reuse existing checkpoints
python run.py --config config/federated.yaml --skip-pgd           # skip PGD step
python run.py --config config/federated.yaml --skip-ratio-sweep   # skip ratio sweep + threshold tuning
python run.py --config config/federated.yaml --centralized        # centralized baseline (no FL)
```

Local / macOS-friendly config example: `config/federated_local_sky.yaml`.

---

## Main pipeline (`run.py`)

| Step | What runs | Main outputs |
|------|-----------|--------------|
| 1 | `scripts/train.py` (or centralized trainer) | `models/global_model.h5`, Flower logs |
| 2 | `compression.py` | `models/tflite/saved_model_*.tflite` |
| 3 | `scripts/analyze_compression.py` | `data/processed/runs/<version>/<run_id>/analysis/` (CSV, JSON, MD, plots) |
| 3b | `scripts/run_pgd.py` | `run_dir/pgd/pgd_report.md`, `pgd_results.json` (skipped if `--skip-pgd` or no `models/global_model.h5`) |
| 4 | `scripts/evaluate_ratio_sweep.py` | `run_dir/eval/ratio_sweep_report.md` |
| 4b | `scripts/tune_threshold_all_ratios.py` | appended to ratio sweep report |
| 5 | `scripts/visualize_results.py` | metric plots under the run directory |

Run root: `data/processed/runs/<version>/<run_id>/`

```
data/processed/runs/<version>/<run_id>/
├── run_config.yaml
├── experiment_record.md
├── models/                 # copied / referenced Keras + TFLite
├── analysis/               # compression analysis
├── pgd/                    # PGD report + JSON (if enabled)
└── eval/                   # ratio sweep + threshold tuning
```

---

## Compression grid sweep + PGD (optional)

For a full **48-combination** table (FL QAT × distillation × pruning × PTQ), PGD on all produced models, and merged CSV:

```bash
python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml --version sweep_pgd
# Quick smoke test (4 combinations):
python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml --quick
```

Outputs include `sweep_compression_grid.csv`, `sweep_compression_grid_with_pgd.csv`, `pgd/pgd_report.md`, and optional `sweep_heatmap_*.png`. See `docs/COMPRESSION_GRID_SWEEP.md`.

---

## Adversarial training (optional)

If `adversarial_training.enabled: true` in your YAML, **adversarial fine-tuning** after FL (FGSM or **PGD** per `adversarial_training.attack`) can be run via `scripts/run_at_after_training.py`. **Evaluation** in the main pipeline uses `run_pgd.py`, whose attack type follows the same `adversarial_training.attack` field (typically **PGD** with `pgd_steps`, `epsilon`, optional `pgd_alpha`).

Legacy one-off FGSM testing:

```bash
python scripts/test_fgsm_attack.py --model path/to/global_model.h5 --config config/federated.yaml
```

---

## Configuration

| File | Role |
|------|------|
| `config/federated.yaml` | Default FL + data + evaluation + compression knobs |
| `config/federated_local.yaml` / `federated_local_sky.yaml` | Local paths, `version` string for run folders |
| `config/fgsm.yaml` | PGD/FGSM **evaluation** defaults: ε grid, subset sizes, `prediction_threshold` |

Important keys:

- **`evaluation.prediction_threshold`** — must match training/eval (e.g. `0.3` for binary IDS).
- **`evaluation.ratio_sweep_models`** — TFLite paths for ratio sweep and PGD model list.
- **`evaluation.pgd_top_n`** — if `> 0`, PGD uses top-N TFLite models from `compression_analysis.json` by `pgd_metric`.
- **`federated.use_qat`** — QAT during FL when supported by the training path.
- **`adversarial_training`** — AT and PGD/FGSM parameters for post-training AT and for `run_pgd.py`.

---

## Repository layout

```
TinyML-friend/
├── run.py                      # Main pipeline
├── compression.py              # Compression → TFLite
├── config/                     # YAML configs
├── data/raw/                   # CIC-IDS2017, Bot-IoT (user-provided)
├── data/processed/runs/        # Run outputs per version / timestamp
├── docs/                       # Notes (e.g. compression grid sweep)
├── esp32_tflite_project/       # ESP32 + TFLite Micro (PlatformIO)
├── scripts/
│   ├── train.py
│   ├── analyze_compression.py
│   ├── run_pgd.py              # PGD (or FGSM if attack=fgsm in config)
│   ├── run_sweep_and_pgd.py    # 48-sweep + PGD + merge + heatmaps
│   ├── sweep_compression_grid.py
│   ├── merge_sweep_pgd.py
│   ├── evaluate_ratio_sweep.py
│   ├── tune_threshold_all_ratios.py
│   ├── visualize_results.py
│   ├── run_at_after_training.py
│   ├── test_fgsm_attack.py     # standalone FGSM test
│   └── …
├── src/
│   ├── adversarial/            # fgsm_hook.py (FGSM + PGD helpers)
│   ├── data/
│   ├── federated/
│   ├── modelcompression/
│   └── …
└── README.md
```

---

## Environment setup

1. **Python venv**

   ```bash
   make setup
   source .venv/bin/activate
   ```

2. **Data**

   - **CIC-IDS2017:** place dataset CSVs under the path given in config (e.g. `data/raw/MachineLearningCVE/`).
   - **Bot-IoT:** optional; see `make download-data` or project docs.

3. **GPU / TensorFlow**

   Use a TensorFlow/Keras stack compatible with the repo (see `requirements.txt`). Some scripts set `TF_USE_LEGACY_KERAS=1` for Keras 2 / `tf_keras` compatibility.

---

## Federated training (standalone)

```bash
python scripts/train.py --config config/federated.yaml
```

- Clients: `src/federated/client.py` (focal loss, class weights, optional QAT).
- Server: `src/federated/server.py` (FedAvg / FedAvgM, LR schedule).

---

## Microcontroller deployment

```bash
python scripts/create_test_model.py
python scripts/deploy_microcontroller.py
python scripts/test_tflite_inference.py
```

ESP32: `esp32_tflite_project/` (PlatformIO). See `docs/MICROCONTROLLER_DEPLOYMENT.md` if present.

---

## Remote server (e.g. Vast.ai)

Sync the project folder to the server workspace (e.g. `/workspace/TinyML`); no special repo layout required. Optional docs: `docs/SERVER_SETUP.md`, `docs/SERVER_SYNC_COMMANDS.md`.

---

## Troubleshooting

- **Load errors / custom loss:** load Keras with `compile=False`, then `compile` for evaluation.
- **F1 / precision / recall = 0:** align `prediction_threshold` across `run_config.yaml`, analysis, and PGD (`config/fgsm.yaml` attack section).
- **Missing dataset:** check `data:` paths in the active YAML.
- **OOM:** lower `max_samples` in config.
- **PGD skipped:** ensure `models/global_model.h5` exists after training.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use this code in academic work, please cite the associated publication and this repository URL. **Authors:** Younsoo Park, Seokhyeon Bae. Supervisor and institution credits are listed at the top of this README.
