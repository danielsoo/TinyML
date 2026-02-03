# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v4 |
| **Run (datetime)** | 2026-01-30_19-14-12 |
| **Data Version** | cicids2017 |
| **Generated** | 2026-01-30 19:15:49 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.2056 | 53,844 | 0.3331 | 0.3038 | 0.46 |
| Compressed | 0.0206 | 18,772 | 0.1710 | 0.2913 | 0.67 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -89.96% ↓ | -16.2170% ↓ | -1.2572% ↓ | +44.57% ↑ | ⚠️ Mixed Results |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed | 9.96x | 90.0% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.2056 MB (215,612 bytes)
- **Parameters**: 53,844
- **Accuracy**: 0.3331
- **Precision**: 0.1848
- **Recall**: 0.8541
- **F1-Score**: 0.3038
- **Avg Latency**: 0.46 ms
- **Samples/sec**: 217321.45
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0206 MB (21,648 bytes)
- **Parameters**: 18,772
- **Accuracy**: 0.1710
- **Precision**: 0.1705
- **Recall**: 0.9998
- **F1-Score**: 0.2913
- **Avg Latency**: 0.67 ms
- **Samples/sec**: 150322.70
- **Compression Ratio**: 9.96x

