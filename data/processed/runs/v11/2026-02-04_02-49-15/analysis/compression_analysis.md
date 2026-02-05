# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-04_02-49-15 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-04 02:49:40 |

## Summary

Total stages analyzed: 2

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.9010 | 0.6415 | 0.9857 | 0.9031 | 1.55 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8242 | 0.0000 | 1.0000 | 0.8242 | 0.41 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -7.6803% ↓ | -64.1533% ↓ | -73.41% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.9010
- **Precision**: 0.8827
- **Recall**: 0.5039
- **F1-Score**: 0.6415
- **Normal Recall** (of actual normal, % predicted as normal): 0.9857
- **Normal Precision** (of predicted normal, % actually normal): 0.9031
- **Avg Latency**: 1.55 ms
- **Samples/sec**: 64432.59
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8242
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, % predicted as normal): 1.0000
- **Normal Precision** (of predicted normal, % actually normal): 0.8242
- **Avg Latency**: 0.41 ms
- **Samples/sec**: 242361.26
- **Compression Ratio**: 11.59x

