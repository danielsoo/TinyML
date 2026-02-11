# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v19 |
| **Run (datetime)** | 2026-02-10_21-15-13 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-10 21:16:19 |

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
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.5 |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 6

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.7371 | 0.4851 | 0.7456 | 0.9195 | 2.27 |
| QAT+Prune only | 0.2367 | 61,969 | 0.9190 | 0.8116 | 0.9054 | 0.9958 | 0.83 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9189 | 0.8107 | 0.9062 | 0.9948 | 0.54 |
| QAT+PTQ | 0.0676 | 61,969 | 0.9189 | 0.8107 | 0.9062 | 0.9948 | 0.54 |
| Compressed (QAT) | 0.0633 | 62,048 | 0.9045 | 0.6361 | 0.9982 | 0.8972 | 0.55 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.9094 | 0.7946 | 0.8925 | 0.9969 | 0.54 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +18.1943% ↑ | +32.6430% ↑ | -63.45% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +18.1854% ↑ | +32.5595% ↑ | -76.40% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +18.1854% ↑ | +32.5595% ↑ | -76.44% ↓ | ✅ All Improved |
| Compressed (QAT) | -91.92% ↓ | +16.7370% ↑ | +15.0974% ↑ | -75.71% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +17.2267% ↑ | +30.9406% ↑ | -76.30% ↓ | ✅ All Improved |

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
| QAT+PTQ | 11.59x | 91.4% |
| Compressed (QAT) | 12.38x | 91.9% |
| noQAT+PTQ | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,632 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.7371
- **Precision**: 0.3719
- **Recall**: 0.6977
- **F1-Score**: 0.4851
- **Normal Recall** (of actual normal, % predicted as normal): 0.7456
- **Normal Precision** (of predicted normal, % actually normal): 0.9195
- **Avg Latency**: 2.27 ms
- **Samples/sec**: 43969.60
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9190
- **Precision**: 0.6915
- **Recall**: 0.9822
- **F1-Score**: 0.8116
- **Normal Recall** (of actual normal, % predicted as normal): 0.9054
- **Normal Precision** (of predicted normal, % actually normal): 0.9958
- **Avg Latency**: 0.83 ms
- **Samples/sec**: 120297.83
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9189
- **Precision**: 0.6924
- **Recall**: 0.9779
- **F1-Score**: 0.8107
- **Normal Recall** (of actual normal, % predicted as normal): 0.9062
- **Normal Precision** (of predicted normal, % actually normal): 0.9948
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 186272.77
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9189
- **Precision**: 0.6924
- **Recall**: 0.9779
- **F1-Score**: 0.8107
- **Normal Recall** (of actual normal, % predicted as normal): 0.9062
- **Normal Precision** (of predicted normal, % actually normal): 0.9948
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 186662.39
- **Compression Ratio**: 11.59x

### Compressed (QAT)

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0633 MB (66,360 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9045
- **Precision**: 0.9822
- **Recall**: 0.4704
- **F1-Score**: 0.6361
- **Normal Recall** (of actual normal, % predicted as normal): 0.9982
- **Normal Precision** (of predicted normal, % actually normal): 0.8972
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 180999.61
- **Compression Ratio**: 12.38x

### noQAT+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9094
- **Precision**: 0.6648
- **Recall**: 0.9873
- **F1-Score**: 0.7946
- **Normal Recall** (of actual normal, % predicted as normal): 0.8925
- **Normal Precision** (of predicted normal, % actually normal): 0.9969
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 185523.00
- **Compression Ratio**: 11.59x

