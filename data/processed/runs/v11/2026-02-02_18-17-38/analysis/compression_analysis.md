# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-02_18-17-38 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-02 18:18:04 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.9490 | 0.8671 | 1.68 |
| Compressed | 0.0731 | 61,969 | 0.8242 | 0.0000 | 0.73 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -90.67% ↓ | -12.4821% ↓ | -86.7056% ↓ | -56.67% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed | 10.72x | 90.7% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,648 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.9490
- **Precision**: 0.8007
- **Recall**: 0.9454
- **F1-Score**: 0.8671
- **Avg Latency**: 1.68 ms
- **Samples/sec**: 59523.22
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0731 MB (76,640 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8242
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 0.73 ms
- **Samples/sec**: 137378.53
- **Compression Ratio**: 10.72x

