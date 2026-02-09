# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-06_04-07-24 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-06 04:09:16 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.6592 | 173,140 | 0.0214 | 0.0285 | 3.12 |
| Compressed (PTQ) | 0.0592 | 53,844 | 0.0735 | 0.0097 | 1.11 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.02% ↓ | +5.2117% ↑ | -1.8783% ↓ | -64.43% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 11.13x | 91.0% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.6592 MB (691,248 bytes)
- **Parameters**: 173,140
- **Accuracy**: 0.0214
- **Precision**: 0.0173
- **Recall**: 0.0816
- **F1-Score**: 0.0285
- **Avg Latency**: 3.12 ms
- **Samples/sec**: 32025.17
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0592 MB (62,088 bytes)
- **Parameters**: 53,844
- **Accuracy**: 0.0735
- **Precision**: 0.0060
- **Recall**: 0.0258
- **F1-Score**: 0.0097
- **Avg Latency**: 1.11 ms
- **Samples/sec**: 90039.37
- **Compression Ratio**: 11.13x

