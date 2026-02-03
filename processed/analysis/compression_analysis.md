# Compression Analysis Report

Generated: 2026-01-26 14:09:12.341306

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.2056 | 53,844 | 0.0168 | 0.0214 | 2.17 |
| Compressed | 0.0206 | 18,772 | 0.0362 | 0.0697 | 1.90 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -89.96% ↓ | +1.9375% ↑ | +4.8382% ↑ | -12.46% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.0168
- **Precision**: 0.0111
- **Recall**: 0.2969
- **F1-Score**: 0.0214
- **Avg Latency**: 2.17 ms
- **Samples/sec**: 46010.86
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0206 MB (21,648 bytes)
- **Parameters**: 18,772
- **Accuracy**: 0.0362
- **Precision**: 0.0361
- **Recall**: 1.0000
- **F1-Score**: 0.0697
- **Avg Latency**: 1.90 ms
- **Samples/sec**: 52558.22
- **Compression Ratio**: 9.96x

