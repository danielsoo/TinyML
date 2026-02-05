# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-04_00-55-09 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-04 00:55:35 |

## Summary

Total stages analyzed: 2

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.9366 | 0.8227 | 0.9580 | 0.9649 | 1.60 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9220 | 0.7300 | 0.9907 | 0.9207 | 0.40 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -1.4622% ↓ | -9.2713% ↓ | -74.94% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.9366
- **Precision**: 0.8093
- **Recall**: 0.8365
- **F1-Score**: 0.8227
- **Normal Recall** (of actual normal, % predicted as normal): 0.9580
- **Normal Precision** (of predicted normal, % actually normal): 0.9649
- **Avg Latency**: 1.60 ms
- **Samples/sec**: 62616.51
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9220
- **Precision**: 0.9323
- **Recall**: 0.5998
- **F1-Score**: 0.7300
- **Normal Recall** (of actual normal, % predicted as normal): 0.9907
- **Normal Precision** (of predicted normal, % actually normal): 0.9207
- **Avg Latency**: 0.40 ms
- **Samples/sec**: 249898.95
- **Compression Ratio**: 11.59x

