# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-06_15-23-59 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-06 15:25:49 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.5972 | 0.4536 | 3.55 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8495 | 0.6911 | 0.86 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +25.2239% ↑ | +23.7541% ↑ | -75.82% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.5972
- **Precision**: 0.2979
- **Recall**: 0.9505
- **F1-Score**: 0.4536
- **Avg Latency**: 3.55 ms
- **Samples/sec**: 28155.93
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8495
- **Precision**: 0.5407
- **Recall**: 0.9575
- **F1-Score**: 0.6911
- **Avg Latency**: 0.86 ms
- **Samples/sec**: 116424.36
- **Compression Ratio**: 11.59x

