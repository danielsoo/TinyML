# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-06_23-25-39 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-06 23:30:28 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.4078 | 0.3557 | 1.87 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9059 | 0.7435 | 1.49 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +49.8071% ↑ | +38.7867% ↑ | -20.23% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.4078
- **Precision**: 0.2183
- **Recall**: 0.9593
- **F1-Score**: 0.3557
- **Avg Latency**: 1.87 ms
- **Samples/sec**: 53508.33
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9059
- **Precision**: 0.6938
- **Recall**: 0.8009
- **F1-Score**: 0.7435
- **Avg Latency**: 1.49 ms
- **Samples/sec**: 67077.74
- **Compression Ratio**: 11.59x

