# Compression Analysis Report

| Item | Value |
|------|----|
| **Run Version** | v1 |
| **Data Version** | cicids2017_max200k_bal5.0 |
| **Generated** | 2026-01-30 14:34:39 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.2056 | 53,844 | 0.2321 | 0.0545 | 0.44 |
| Compressed | 0.0206 | 18,772 | 0.0385 | 0.0699 | 0.65 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -89.96% ↓ | -19.3575% ↓ | +1.5350% ↑ | +46.55% ↑ | ✅ Mostly Better |

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
- **Accuracy**: 0.2321
- **Precision**: 0.0285
- **Recall**: 0.6131
- **F1-Score**: 0.0545
- **Avg Latency**: 0.44 ms
- **Samples/sec**: 225282.20
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0206 MB (21,648 bytes)
- **Parameters**: 18,772
- **Accuracy**: 0.0385
- **Precision**: 0.0362
- **Recall**: 1.0000
- **F1-Score**: 0.0699
- **Avg Latency**: 0.65 ms
- **Samples/sec**: 153721.97
- **Compression Ratio**: 9.96x

