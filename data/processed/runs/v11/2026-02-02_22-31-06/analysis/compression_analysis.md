# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-02_22-31-06 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-02 22:31:31 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.7698 | 0.5693 | 1.60 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8257 | 0.0223 | 0.41 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +5.5884% ↑ | -54.6939% ↓ | -74.60% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,648 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.7698
- **Precision**: 0.4242
- **Recall**: 0.8652
- **F1-Score**: 0.5693
- **Avg Latency**: 1.60 ms
- **Samples/sec**: 62491.49
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8257
- **Precision**: 0.8066
- **Recall**: 0.0113
- **F1-Score**: 0.0223
- **Avg Latency**: 0.41 ms
- **Samples/sec**: 246029.09
- **Compression Ratio**: 11.59x

