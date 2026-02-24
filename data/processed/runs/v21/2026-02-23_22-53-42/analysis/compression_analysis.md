# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v21 |
| **Run (datetime)** | 2026-02-23_22-53-42 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-23 22:54:45 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 2000000 |
| **Balance ratio** (normal:attack) | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 2 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.7 |
| **Use distillation** | False |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **LR decay type** | cosine |
| **LR decay rate** | 0.97 |
| **LR drop rate** | 0.5 |
| **LR epochs drop** | 5 |
| **LR min** | 0.0001 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |
| **Min available clients** | 4 |
| **Prediction threshold** | 0.3 |
| **Ratio sweep models** | 6 models |
| **Always build traditional** | True |
| **Traditional model path** | null |

## Summary

Total stages analyzed: 6

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7835 | 205,777 | 0.9155 | 0.7829 | 0.9279 | 0.9680 | 1.87 |
| QAT+Prune only | 0.2367 | 61,969 | 0.9318 | 0.7821 | 0.9840 | 0.9363 | 0.76 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9319 | 0.7825 | 0.9840 | 0.9364 | 0.49 |
| QAT+PTQ | 0.0676 | 61,969 | 0.9319 | 0.7825 | 0.9840 | 0.9364 | 0.48 |
| Compressed (QAT) | 0.0635 | 62,048 | 0.8906 | 0.6103 | 0.9786 | 0.8976 | 0.48 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.2204 | 0.3129 | 0.0521 | 1.0000 | 0.48 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +1.6230% ↑ | -0.0845% ↓ | -59.30% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.38% ↓ | +1.6339% ↑ | -0.0444% ↓ | -73.81% ↓ | ✅ Mostly Better |
| QAT+PTQ | -91.38% ↓ | +1.6339% ↑ | -0.0444% ↓ | -74.45% ↓ | ✅ Mostly Better |
| Compressed (QAT) | -91.89% ↓ | -2.4977% ↓ | -17.2587% ↓ | -74.38% ↓ | ✅ Mostly Better |
| noQAT+PTQ | -91.38% ↓ | -69.5135% ↓ | -47.0012% ↓ | -74.49% ↓ | ✅ Mostly Better |

## Pipeline overview (How each stage is produced)

| Stage | Input | Processing | Output file |
|-------|-------|------------|-------------|
| **Keras** | FL/central training done | Use as-is (no QAT strip) or load .h5 | `models/global_model.h5` |
| **Original (TFLite)** | Keras model | float32 TFLite export (no quantization) | `saved_model_original.tflite` |
| **QAT+Prune only** | QAT-trained model | QAT strip → 50% structured pruning → **float32** TFLite (no quant) | `saved_model_qat_pruned_float32.tflite` |
| **Compressed (QAT)** | QAT-trained model | QAT strip → 50% prune → **quantize_model** → 2 epoch fine-tune → **int8** TFLite | `saved_model_pruned_qat.tflite` |
| **QAT+PTQ** | QAT-trained model | QAT strip → 50% prune → **PTQ** (quantize with representative data) → int8 TFLite | `saved_model_qat_ptq.tflite` |
| **noQAT+PTQ** | Model trained **without QAT** | 50% prune → PTQ → int8 TFLite | `saved_model_no_qat_ptq.tflite` |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| QAT+Prune only | 3.31x | 69.8% |
| Compressed (PTQ) | 11.59x | 91.4% |
| QAT+PTQ | 11.60x | 91.4% |
| Compressed (QAT) | 12.34x | 91.9% |
| noQAT+PTQ | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7835 MB (821,560 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.9155
- **Precision**: 0.7199
- **Recall**: 0.8581
- **F1-Score**: 0.7829
- **Normal Recall** (of actual normal, % predicted as normal): 0.9279
- **Normal Precision** (of predicted normal, % actually normal): 0.9680
- **Avg Latency**: 1.87 ms
- **Samples/sec**: 53610.92
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,156 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9318
- **Precision**: 0.9030
- **Recall**: 0.6898
- **F1-Score**: 0.7821
- **Normal Recall** (of actual normal, % predicted as normal): 0.9840
- **Normal Precision** (of predicted normal, % actually normal): 0.9363
- **Avg Latency**: 0.76 ms
- **Samples/sec**: 131714.11
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,856 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9319
- **Precision**: 0.9032
- **Recall**: 0.6903
- **F1-Score**: 0.7825
- **Normal Recall** (of actual normal, % predicted as normal): 0.9840
- **Normal Precision** (of predicted normal, % actually normal): 0.9364
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 204700.05
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,848 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9319
- **Precision**: 0.9032
- **Recall**: 0.6903
- **F1-Score**: 0.7825
- **Normal Recall** (of actual normal, % predicted as normal): 0.9840
- **Normal Precision** (of predicted normal, % actually normal): 0.9364
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 209788.63
- **Compression Ratio**: 11.60x

### Compressed (QAT)

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.8906
- **Precision**: 0.8294
- **Recall**: 0.4828
- **F1-Score**: 0.6103
- **Normal Recall** (of actual normal, % predicted as normal): 0.9786
- **Normal Precision** (of predicted normal, % actually normal): 0.8976
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 209223.52
- **Compression Ratio**: 12.34x

### noQAT+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,856 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.2204
- **Precision**: 0.1855
- **Recall**: 1.0000
- **F1-Score**: 0.3129
- **Normal Recall** (of actual normal, % predicted as normal): 0.0521
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 210124.94
- **Compression Ratio**: 11.59x

