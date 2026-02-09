# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-07_11-51-08 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-07 11:55:37 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.6566 | 0.4831 | 3.52 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9054 | 0.7534 | 0.88 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +24.8785% ↑ | +27.0270% ↑ | -74.95% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.6566
- **Precision**: 0.3249
- **Recall**: 0.9419
- **F1-Score**: 0.4831
- **Avg Latency**: 3.52 ms
- **Samples/sec**: 28414.96
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9054
- **Precision**: 0.6779
- **Recall**: 0.8479
- **F1-Score**: 0.7534
- **Avg Latency**: 0.88 ms
- **Samples/sec**: 113451.56
- **Compression Ratio**: 11.59x

