# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v16 |
| **Run (datetime)** | 2026-02-09_01-17-11 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-09 01:18:59 |

## Summary

Total stages analyzed: 3

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Keras | 0.8439 | 208,385 | 0.8296 | 0.0000 | 157.91 |
| Original (TFLite) | 0.7836 | 205,777 | 0.7836 | 0.4187 | 2.26 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9092 | 0.7473 | 0.55 |

## 📊 Improvements vs Baseline

**Baseline:** Keras

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Keras | - | - | - | - | **Baseline** |
| Original (TFLite) | -7.14% ↓ | -4.6050% ↓ | +41.8743% ↑ | -98.57% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.99% ↓ | +7.9590% ↑ | +74.7318% ↑ | -99.65% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.8296
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 157.91 ms
- **Samples/sec**: 633.28
- **Compression Ratio**: 1.00x

### Original (TFLite)

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,652 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.7836
- **Precision**: 0.3860
- **Recall**: 0.4576
- **F1-Score**: 0.4187
- **Avg Latency**: 2.26 ms
- **Samples/sec**: 44306.81
- **Compression Ratio**: 1.08x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9092
- **Precision**: 0.7107
- **Recall**: 0.7880
- **F1-Score**: 0.7473
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 183117.40
- **Compression Ratio**: 12.48x

