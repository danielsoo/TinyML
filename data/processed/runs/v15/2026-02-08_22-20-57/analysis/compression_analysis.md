# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v15 |
| **Run (datetime)** | 2026-02-08_22-20-57 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-08 22:21:54 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 1500000 |
| **Balance ratio** (normal:attack) | 4.0 |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.75 |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 3

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Keras | 0.8439 | 208,385 | 0.8241 | 0.0000 | 153.09 |
| Original (TFLite) | 0.7836 | 205,777 | 0.7895 | 0.4946 | 2.20 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8577 | 0.6930 | 0.56 |

## 📊 Improvements vs Baseline

**Baseline:** Keras

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Keras | - | - | - | - | **Baseline** |
| Original (TFLite) | -7.14% ↓ | -3.4587% ↓ | +49.4552% ↑ | -98.56% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.99% ↓ | +3.3622% ↑ | +69.2966% ↑ | -99.63% ↓ | ✅ All Improved |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Keras | 1.00x | 0.0% |
| Original (TFLite) | 1.08x | 7.1% |
| Compressed (PTQ) | 12.48x | 92.0% |

## Detailed Metrics

### Keras

- **Model Path**: `models/global_model.h5`
- **File Size**: 0.8439 MB (884,848 bytes)
- **Parameters**: 208,385
- **Accuracy**: 0.8241
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 153.09 ms
- **Samples/sec**: 653.21
- **Compression Ratio**: 1.00x

### Original (TFLite)

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,652 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.7895
- **Precision**: 0.4281
- **Recall**: 0.5854
- **F1-Score**: 0.4946
- **Avg Latency**: 2.20 ms
- **Samples/sec**: 45471.64
- **Compression Ratio**: 1.08x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8577
- **Precision**: 0.5585
- **Recall**: 0.9128
- **F1-Score**: 0.6930
- **Avg Latency**: 0.56 ms
- **Samples/sec**: 178579.81
- **Compression Ratio**: 12.48x

