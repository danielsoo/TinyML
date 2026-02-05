# Compression Analysis Report

| 항목 | 값 |
|------|----|
| **Version** | v8_centralized |
| **Run (날짜_시간)** | 2026-02-02_00-48-48 |
| **Data Version** | cicids2017_max1500k |
| **Generated** | 2026-02-02 00:51:13 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.2056 | 53,844 | 0.3468 | 0.3527 | 0.70 |
| Compressed | 0.0206 | 18,772 | 0.2162 | 0.3538 | 1.00 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -89.96% ↓ | -13.0623% ↓ | +0.1137% ↑ | +42.16% ↑ | ✅ Mostly Better |

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
- **Accuracy**: 0.3468
- **Precision**: 0.2240
- **Recall**: 0.8289
- **F1-Score**: 0.3527
- **Avg Latency**: 0.70 ms
- **Samples/sec**: 142218.36
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0206 MB (21,648 bytes)
- **Parameters**: 18,772
- **Accuracy**: 0.2162
- **Precision**: 0.2149
- **Recall**: 0.9996
- **F1-Score**: 0.3538
- **Avg Latency**: 1.00 ms
- **Samples/sec**: 100040.64
- **Compression Ratio**: 9.96x

