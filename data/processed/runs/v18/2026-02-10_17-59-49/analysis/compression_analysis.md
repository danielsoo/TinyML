# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v18 |
| **Run (datetime)** | 2026-02-10_17-59-49 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-10 18:01:13 |

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
| **FL rounds** | 1 |
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

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Keras | 0.8279 | 204,826 | 0.2848 | 0.3317 | 77.16 |
| Original (TFLite) | 0.7834 | 205,777 | 0.2660 | 0.3260 | 2.30 |
| QAT+일반압축 | 0.2367 | 61,969 | 0.1816 | 0.3025 | 0.86 |
| Compressed (QAT) | 0.0633 | 62,048 | 0.9410 | 0.8532 | 0.54 |
| QAT+PTQ | 0.0676 | 61,969 | 0.1816 | 0.3025 | 0.55 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.1776 | 0.3015 | 0.55 |

## 📊 Improvements vs Baseline

**Baseline:** Keras

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Keras | - | - | - | - | **Baseline** |
| Original (TFLite) | -5.36% ↓ | -1.8850% ↓ | -0.5742% ↓ | -97.03% ↓ | ✅ Mostly Better |
| QAT+일반압축 | -71.41% ↓ | -10.3247% ↓ | -2.9195% ↓ | -98.88% ↓ | ✅ Mostly Better |
| Compressed (QAT) | -92.36% ↓ | +65.6201% ↑ | +52.1438% ↑ | -99.30% ↓ | ✅ All Improved |
| QAT+PTQ | -91.84% ↓ | -10.3210% ↓ | -2.9185% ↓ | -99.28% ↓ | ✅ Mostly Better |
| noQAT+PTQ | -91.83% ↓ | -10.7247% ↓ | -3.0168% ↓ | -99.29% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Keras | 1.00x | 0.0% |
| Original (TFLite) | 1.06x | 5.4% |
| QAT+일반압축 | 3.50x | 71.4% |
| Compressed (QAT) | 13.08x | 92.4% |
| QAT+PTQ | 12.25x | 91.8% |
| noQAT+PTQ | 12.24x | 91.8% |

## Detailed Metrics

### Keras

- **Model Path**: `models/global_model.h5`
- **File Size**: 0.8279 MB (868,072 bytes)
- **Parameters**: 204,826
- **Accuracy**: 0.2848
- **Precision**: 0.1988
- **Recall**: 0.9998
- **F1-Score**: 0.3317
- **Avg Latency**: 77.16 ms
- **Samples/sec**: 1295.97
- **Compression Ratio**: 1.00x

### Original (TFLite)

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7834 MB (821,504 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.2660
- **Precision**: 0.1947
- **Recall**: 0.9998
- **F1-Score**: 0.3260
- **Avg Latency**: 2.30 ms
- **Samples/sec**: 43570.85
- **Compression Ratio**: 1.06x

### QAT+일반압축

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,200 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.1816
- **Precision**: 0.1782
- **Recall**: 0.9997
- **F1-Score**: 0.3025
- **Avg Latency**: 0.86 ms
- **Samples/sec**: 115618.82
- **Compression Ratio**: 3.50x

### Compressed (QAT)

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0633 MB (66,360 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9410
- **Precision**: 0.7647
- **Recall**: 0.9648
- **F1-Score**: 0.8532
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 184203.07
- **Compression Ratio**: 13.08x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,872 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.1816
- **Precision**: 0.1782
- **Recall**: 0.9997
- **F1-Score**: 0.3025
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 181109.03
- **Compression Ratio**: 12.25x

### noQAT+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.1776
- **Precision**: 0.1775
- **Recall**: 1.0000
- **F1-Score**: 0.3015
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 182131.40
- **Compression Ratio**: 12.24x

