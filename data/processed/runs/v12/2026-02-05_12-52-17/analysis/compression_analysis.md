# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-05_12-52-17 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-05 12:54:08 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.8275 | 0.6487 | 3.10 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8806 | 0.7437 | 0.79 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +5.3119% ↑ | +9.4962% ↑ | -74.48% ↓ | ✅ All Improved |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,632 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.8275
- **Precision**: 0.5054
- **Recall**: 0.9055
- **F1-Score**: 0.6487
- **Avg Latency**: 3.10 ms
- **Samples/sec**: 32256.19
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8806
- **Precision**: 0.5975
- **Recall**: 0.9846
- **F1-Score**: 0.7437
- **Avg Latency**: 0.79 ms
- **Samples/sec**: 126399.18
- **Compression Ratio**: 11.59x

