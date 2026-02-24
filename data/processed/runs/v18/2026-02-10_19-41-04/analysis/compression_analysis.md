# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v18 |
| **Run (datetime)** | 2026-02-10_19-41-04 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-10 19:42:27 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 2000000 |
| **Balance ratio** (normal:attack) | 4.0 |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **Aggregation strategy** | FedAvgM (momentum=0.5) |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.5 |
| **Prediction threshold** | 0.15 |
| **Use QAT** | False |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 6

## Pipeline overview (How each stage is produced)

| Stage | Input | Processing | Output file |
|-------|-------|------------|-------------|
| **Keras** | FL/central training done | Use as-is (no QAT strip) or load .h5 | `models/global_model.h5` |
| **Original (TFLite)** | Keras model | float32 TFLite export (no quantization) | `saved_model_original.tflite` |
| **QAT+Prune only** | QAT-trained model | QAT strip → 50% structured pruning → **float32** TFLite (no quant) | `saved_model_qat_pruned_float32.tflite` |
| **Compressed (QAT)** | QAT-trained model | QAT strip → 50% prune → **quantize_model** → 2 epoch fine-tune → **int8** TFLite | `saved_model_pruned_qat.tflite` |
| **QAT+PTQ** | QAT-trained model | QAT strip → 50% prune → **PTQ** (quantize with representative data) → int8 TFLite | `saved_model_qat_ptq.tflite` |
| **noQAT+PTQ** | Model trained **without QAT** | 50% prune → PTQ → int8 TFLite | `saved_model_no_qat_ptq.tflite` |

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Keras | 0.8279 | 204,826 | 0.8296 | 0.2910 | 77.99 |
| Original (TFLite) | 0.7834 | 205,777 | 0.3013 | 0.3198 | 2.22 |
| QAT+Prune only | 0.2367 | 61,969 | 0.2144 | 0.3111 | 0.82 |
| Compressed (QAT) | 0.0633 | 62,048 | 0.6012 | 0.4601 | 0.52 |
| QAT+PTQ | 0.0676 | 61,969 | 0.2140 | 0.3110 | 0.52 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.1776 | 0.3015 | 0.52 |

## 📊 Improvements vs Baseline

**Baseline:** Keras

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Keras | - | - | - | - | **Baseline** |
| Original (TFLite) | -5.36% ↓ | -52.8255% ↓ | +2.8833% ↑ | -97.15% ↓ | ✅ Mostly Better |
| QAT+Prune only | -71.41% ↓ | -61.5202% ↓ | +2.0190% ↑ | -98.95% ↓ | ✅ Mostly Better |
| Compressed (QAT) | -92.36% ↓ | -22.8390% ↓ | +16.9112% ↑ | -99.34% ↓ | ✅ Mostly Better |
| QAT+PTQ | -91.84% ↓ | -61.5580% ↓ | +2.0072% ↑ | -99.33% ↓ | ✅ Mostly Better |
| noQAT+PTQ | -91.83% ↓ | -65.1982% ↓ | +1.0592% ↑ | -99.33% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Keras | 1.00x | 0.0% |
| Original (TFLite) | 1.06x | 5.4% |
| QAT+Prune only | 3.50x | 71.4% |
| Compressed (QAT) | 13.08x | 92.4% |
| QAT+PTQ | 12.25x | 91.8% |
| noQAT+PTQ | 12.24x | 91.8% |

## Detailed Metrics

### Keras

- **Model Path**: `models/global_model.h5`
- **File Size**: 0.8279 MB (868,072 bytes)
- **Parameters**: 204,826
- **Accuracy**: 0.8296
- **Precision**: 0.5565
- **Recall**: 0.1970
- **F1-Score**: 0.2910
- **Avg Latency**: 77.99 ms
- **Samples/sec**: 1282.17
- **Compression Ratio**: 1.00x

### Original (TFLite)

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7834 MB (821,504 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.3013
- **Precision**: 0.1933
- **Recall**: 0.9251
- **F1-Score**: 0.3198
- **Avg Latency**: 2.22 ms
- **Samples/sec**: 45015.34
- **Compression Ratio**: 1.06x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,200 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.2144
- **Precision**: 0.1843
- **Recall**: 0.9994
- **F1-Score**: 0.3111
- **Avg Latency**: 0.82 ms
- **Samples/sec**: 122507.93
- **Compression Ratio**: 3.50x

### Compressed (QAT)

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0633 MB (66,360 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.6012
- **Precision**: 0.3028
- **Recall**: 0.9571
- **F1-Score**: 0.4601
- **Avg Latency**: 0.52 ms
- **Samples/sec**: 193125.70
- **Compression Ratio**: 13.08x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,872 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.2140
- **Precision**: 0.1842
- **Recall**: 0.9993
- **F1-Score**: 0.3110
- **Avg Latency**: 0.52 ms
- **Samples/sec**: 192372.79
- **Compression Ratio**: 12.25x

### noQAT+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.1776
- **Precision**: 0.1775
- **Recall**: 1.0000
- **F1-Score**: 0.3015
- **Avg Latency**: 0.52 ms
- **Samples/sec**: 190875.76
- **Compression Ratio**: 12.24x

