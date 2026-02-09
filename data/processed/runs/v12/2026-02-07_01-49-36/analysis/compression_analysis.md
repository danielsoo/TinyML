# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-07_01-49-36 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-07 01:50:18 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.3874 | 0.3168 | 1.87 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8925 | 0.7385 | 0.51 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +50.5133% ↑ | +42.1689% ↑ | -72.64% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.3874
- **Precision**: 0.1970
- **Recall**: 0.8075
- **F1-Score**: 0.3168
- **Avg Latency**: 1.87 ms
- **Samples/sec**: 53342.97
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8925
- **Precision**: 0.6454
- **Recall**: 0.8629
- **F1-Score**: 0.7385
- **Avg Latency**: 0.51 ms
- **Samples/sec**: 194966.02
- **Compression Ratio**: 11.59x

