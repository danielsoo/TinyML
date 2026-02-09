# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-08_01-00-49 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-08 01:05:34 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.6153 | 0.4281 | 4.02 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9343 | 0.8146 | 0.98 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +31.8924% ↑ | +38.6428% ↑ | -75.65% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.6153
- **Precision**: 0.2867
- **Recall**: 0.8452
- **F1-Score**: 0.4281
- **Avg Latency**: 4.02 ms
- **Samples/sec**: 24899.99
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9343
- **Precision**: 0.7841
- **Recall**: 0.8475
- **F1-Score**: 0.8146
- **Avg Latency**: 0.98 ms
- **Samples/sec**: 102277.65
- **Compression Ratio**: 11.59x

