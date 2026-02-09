# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v15 |
| **Run (datetime)** | 2026-02-08_19-41-54 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-08 19:42:36 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 1500000 |
| **Balance ratio** (normal:attack) | 4.0 |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.75 |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 2

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7839 | 205,777 | 0.8241 | 0.0000 | 2.41 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9018 | 0.7786 | 0.55 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.38% ↓ | +7.7640% ↑ | +77.8645% ↑ | -77.36% ↓ | ✅ All Improved |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 11.60x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7839 MB (821,948 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.8241
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 2.41 ms
- **Samples/sec**: 41515.02
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,872 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9018
- **Precision**: 0.6449
- **Recall**: 0.9824
- **F1-Score**: 0.7786
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 183341.52
- **Compression Ratio**: 11.60x

