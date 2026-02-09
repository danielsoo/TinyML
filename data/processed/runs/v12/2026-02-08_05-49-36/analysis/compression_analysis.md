# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-08_05-49-36 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-08 05:54:00 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.8744 | 0.7272 | 6.19 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9234 | 0.7895 | 0.83 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +4.9047% ↑ | +6.2323% ↑ | -86.67% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.8744
- **Precision**: 0.5772
- **Recall**: 0.9825
- **F1-Score**: 0.7272
- **Avg Latency**: 6.19 ms
- **Samples/sec**: 16152.32
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9234
- **Precision**: 0.7427
- **Recall**: 0.8426
- **F1-Score**: 0.7895
- **Avg Latency**: 0.83 ms
- **Samples/sec**: 121149.13
- **Compression Ratio**: 11.59x

