# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v17 |
| **Run (datetime)** | 2026-02-09_17-52-41 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-09 17:53:49 |

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
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.5 |
| **Prediction threshold** | 0.3 |
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
| Keras | 0.8439 | 208,385 | 0.8225 | 0.0000 | 236.84 |
| Original (TFLite) | 0.7836 | 205,777 | 0.6695 | 0.5021 | 2.34 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9439 | 0.8443 | 0.55 |

## 📊 Improvements vs Baseline

**Baseline:** Keras

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Keras | - | - | - | - | **Baseline** |
| Original (TFLite) | -7.14% ↓ | -15.3005% ↓ | +50.2148% ↑ | -99.01% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.99% ↓ | +12.1438% ↑ | +84.4287% ↑ | -99.77% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.8225
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 236.84 ms
- **Samples/sec**: 422.23
- **Compression Ratio**: 1.00x

### Original (TFLite)

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,632 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.6695
- **Precision**: 0.3427
- **Recall**: 0.9390
- **F1-Score**: 0.5021
- **Avg Latency**: 2.34 ms
- **Samples/sec**: 42661.89
- **Compression Ratio**: 1.08x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9439
- **Precision**: 0.8323
- **Recall**: 0.8566
- **F1-Score**: 0.8443
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 181579.46
- **Compression Ratio**: 12.48x

