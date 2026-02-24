# Version Changelog (Detailed)

Detailed per-version change log. For reference in papers/reports.

---

## Common Settings (Baseline)

| Category | Item | Value |
|----------|------|-------|
| **Model** | Architecture | MLP: 512→256→128 (Dense+BN+Dropout) |
| | Dropout rate | 0.25 |
| | BatchNorm | Applied per Dense layer |
| **Data** | Dataset | CIC-IDS2017 (78 features, binary BENIGN/ATTACK) |
| | Preprocessing | StandardScaler, duplicate removal |
| **FL** | num_clients | 4 |
| | batch_size | 128 |
| | learning_rate | 0.001 |
| **Compression** | Pruning type | Structured pruning (neuron removal) |
| | Skip last layer | true |

---

## v1

| Category | Item | Value |
|----------|------|-------|
| **Training** | Dataset | CIC-IDS2017 |
| | max_samples | 200,000 |
| | balance_ratio | 5.0 (majority ≤ 5× minority) |
| | Aggregation | FedAvg |
| | num_rounds | (default) |
| | local_epochs | (default) |
| **Compression** | Pruning ratio | 50% |
| | Quantization | PTQ, Full INT8 (assumed) |
| | TFLite | default export |

**Note**: Initial setup, 78 features (CIC-IDS2017).

---

## v3

| Category | Item | Value |
|----------|------|-------|
| **Training** | max_samples | 1,500,000 |
| | balance_ratio | (unspecified) |
| **Compression** | Same as v1 | prune 50%, INT8, TFLite |

---

## v4

| Category | Item | Value |
|----------|------|-------|
| **Training** | max_samples | No limit (full dataset) |
| | Other | Same as v3 |
| **Issue** | Accuracy drop | Accuracy drop observed with full dataset |

---

## v5

| Category | Item | Value |
|----------|------|-------|
| **Training** | Model | Deeper/wider MLP (512-256-128) |
| | Loss | Focal loss (γ=2.0, α=0.25~0.75) |
| | Optimizer | Adam, clipnorm=1.0 |
| | Callbacks | ReduceLROnPlateau, EarlyStopping |
| | Aggregation | FedAvgM (server_momentum=0.9, server_lr=1.0) |
| | num_rounds | 25 |
| | local_epochs | 15 |
| | use_class_weights | true |
| **Compression** | Same as v1 | prune 50%, INT8, TFLite |

**Note**: Training algorithm improvements, FedAvg→FedAvgM switch.

---

## v6

| Category | Item | Value |
|----------|------|-------|
| **Training** | Same as v5 | Version tag cleanup |
| **Compression** | Same as v1 | - |

---

## v7

| Category | Item | Value |
|----------|------|-------|
| **Training** | Same as v5/v6 | - |
| **Issue** | Accuracy ~12% | FedAvg/FedAvgM behavior review |

---

## v8

| Category | Item | Value |
|----------|------|-------|
| **Training** | balance_ratio | 10.0 (majority ≤ 10× minority) |
| | focal_loss_alpha | 0.75 |
| | use_smote | true |
| | max_samples | 1,500,000 |
| | Option | `--centralized` added (v8_centralized) |
| **Compression** | Pruning | 50%, skip_last_layer |
| | Quantization | PTQ |

---

## v8_centralized

| Category | Item | Value |
|----------|------|-------|
| **Training** | Mode | Centralized (no FedAvg) |
| | Purpose | FL vs Centralized baseline comparison |
| | Config | Same as v8 (data, model) |
| **Compression** | Same as v8 | - |

---

## v11

| Category | Item | Value |
|----------|------|-------|
| **Training** | balance_ratio | **1.0** (50:50 undersample) |
| | focal_loss_alpha | **0.85** (minority emphasis) |
| | use_smote | true |
| | max_samples | 1,500,000 |
| | num_rounds | 25 |
| | local_epochs | 15 |
| | server_momentum | 0.9 |
| | server_learning_rate | 1.0 |
| **Compression** | BN folding | ✅ Applied (before TFLite conversion) |
| | PTQ mode | **Full INT8 → Dynamic Range Quantization** |
| | DRQ description | int8 weights, float32 input/output |
| | QAT | Option (saved_model_pruned_qat.tflite) |
| | Pruning | 50%, skip_last_layer |
| | QAT fine-tune | 2 epochs, batch 128, 5k samples |

### Compression details (v11)

| Item | v1~v8 | v11 |
|------|-------|-----|
| BatchNorm | NaN on TFLite conversion | BN folding → merged into Dense |
| PTQ | Full INT8 (int8 in/out) | DRQ (int8 weights, float32 I/O) |
| Full INT8 issue | P/R/F1=0 (sigmoid precision loss) | DRQ preserves accuracy |
| QAT | Not supported | tfmot.quantize_model, 2 epoch fine-tune |

### v11 example result (2026-02-02_23-28-45)

| Stage | Size (MB) | Params | Accuracy | Precision | Recall | F1 | Latency (ms) |
|-------|-----------|--------|----------|------------|--------|-----|--------------|
| Original | 0.784 | 205,777 | 0.937 | 0.744 | 0.981 | 0.846 | 1.54 |
| Compressed (PTQ) | 0.068 | 61,969 | 0.902 | 0.950 | 0.467 | 0.626 | 0.43 |

- Compression ratio: 11.59×
- Recall drop: 0.98→0.47 after DRQ (room for improvement: QAT, pruning 30%, threshold tuning)

---

## Run-Level Differences (per-run by timestamp)

Multiple runs (timestamps) can exist per version. Code/training results may differ per run.

**Versions with multiple runs**: v8 (2), v8_centralized (2), v11 (5)  
**Single run**: v1, v3, v4, v5, v6, v7

---

### v11 (5 runs, 2026-02-02)

| Run | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig MB | Comp MB | Params | F1(O/C) | Latency(O/C) | Code/Env |
|-----|----------|----------|----------|----------|---------|---------|--------|---------|--------------|----------|
| 16-13-48 | 0.824 | 0.00/0.00 | 0.839 | 1.00/0.08 | 0.784 | 0.073 | 205,777 | 0.00/0.15 | 1.73/0.70 ms | BN strip not applied, Original TFLite NaN |
| 18-17-38 | 0.949 | 0.80/0.95 | 0.824 | 0.00/0.00 | 0.784 | 0.073 | 205,777 | 0.87/0.00 | 1.68/0.73 ms | BN strip applied, Compressed Full INT8 collapse |
| 19-11-40 | 0.949 | 0.80/0.95 | 0.824 | 0.00/0.00 | 0.784 | 0.074 | 205,777 | 0.87/0.00 | - | Same |
| 19-16-09 | 0.949 | 0.80/0.95 | 0.824 | 0.00/0.00 | 0.784 | 0.073 | 205,777 | 0.87/0.00 | - | Same |
| **23-28-45** | **0.937** | **0.74/0.98** | **0.902** | **0.95/0.47** | **0.784** | **0.068** | **205,777** | **0.85/0.63** | **1.54/0.43 ms** | **DRQ applied, final OK** |

**Timeline**:
1. 16-13: BN not folded in Original TFLite → NaN/collapse. Compressed mostly negative predictions (R=0.08).
2. 18-17~19-16: BN folding applied → Original OK. Compressed Full INT8 → P/R=0.
3. 23-28: DRQ applied → Original and Compressed both OK. Comp 70.8KB (was 76.6KB).

### v8 (2 runs, 2026-02-01~02)

| Run | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig MB | Comp MB | Params | F1(Orig) | F1(Comp) | Latency(Orig) | Latency(Comp) |
|-----|----------|----------|----------|----------|---------|---------|--------|----------|----------|----------------|----------------|
| **02-01 20-53-20** | 0.347 | 0.22/0.83 | 0.216 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | 0.353 | 0.354 | 0.45 ms | 0.64 ms |
| 02-02 01-38-52 | 0.785 | 0.00/0.00 | 0.215 | 0.21/1.00 | 0.784 | 0.073 | 205,777 | 0.000 | 0.353 | 1.64 ms | 0.75 ms |

**Timeline**:
1. **20-53-20 (02-01)**: Small model (53k params, 0.2MB, ~38 feat). Original and Compressed both OK P/R. Low Acc (0.35) likely from imbalance (e.g. balance_ratio 10.0).
2. **01-38-52 (02-02)**: Large model (205k, 0.78MB, 78 feat, CIC-IDS2017). Training Acc 0.78 but Original TFLite P/R=0/0 (BN not folded → NaN/collapse). Compressed runs with R=1.0.

**Model difference**: 20-53 is different training/data (small); 01-38 is large structure similar to v11. Same config (v8) but run time and results differ.

### v8_centralized (2 runs, 2026-02-01~02)

| Run | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig MB | Comp MB | Params | F1 | Latency(Orig) | Latency(Comp) |
|-----|----------|----------|----------|----------|---------|---------|--------|-----|---------------|---------------|
| 02-01 22-48-06 | 0.347 | 0.22/0.83 | 0.216 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | 0.353 | 0.45 ms | 0.65 ms |
| 02-02 00-48-48 | 0.347 | 0.22/0.83 | 0.216 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | 0.353 | 0.70 ms | 1.00 ms |

**Timeline**:
1. **22-48-06 (02-01)**: Centralized training, small model. Same model size/metrics as v8 FL 20-53.
2. **00-48-48 (02-02)**: Same config, same metrics. Slight latency difference (measurement env).

**Note**: Both runs same small model, same Acc/P/R/F1. Same structure as v8 FL 20-53, serves as centralized baseline.

### Run-Level doc generation

```bash
python scripts/generate_run_level_changelog.py --analysis-dir <path>
```

→ Generates `RUN_LEVEL_DIFFERENCES.md` in the analysis folder.

---

## Glossary

| Term | Description |
|------|-------------|
| FedAvg | Federated Averaging (weighted average) |
| FedAvgM | FedAvg + server-side momentum |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| DRQ | Dynamic Range Quantization (weights only int8) |
| Full INT8 | weights, activations, I/O all int8 |
| BN folding | BatchNorm mathematically merged into preceding Dense |
| balance_ratio | majority ≤ ratio × minority (undersample) |
| focal_loss_alpha | positive (majority) class weight |
