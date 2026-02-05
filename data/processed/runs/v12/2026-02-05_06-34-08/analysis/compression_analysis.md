# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-05_06-34-08 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-02-05 06:35:57 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | /scratch/yqp5187/Bot-IoT |
| **Max samples** | - |
| **Balance ratio** (정상:공격) | 4.0 (정상:공격 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 50 |
| **Local epochs** | 10 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.85 |
| **Use QAT** | True |
| **Server momentum** | 0.9 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 2

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.8296 | 0.0000 | 1.0000 | 0.8296 | 1.14 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8282 | 0.0012 | 0.9982 | 0.8295 | 0.42 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -0.1398% ↓ | +0.1210% ↑ | -63.20% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.8296
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, % predicted as normal): 1.0000
- **Normal Precision** (of predicted normal, % actually normal): 0.8296
- **Avg Latency**: 1.14 ms
- **Samples/sec**: 87613.14
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8282
- **Precision**: 0.0648
- **Recall**: 0.0006
- **F1-Score**: 0.0012
- **Normal Recall** (of actual normal, % predicted as normal): 0.9982
- **Normal Precision** (of predicted normal, % actually normal): 0.8295
- **Avg Latency**: 0.42 ms
- **Samples/sec**: 238055.74
- **Compression Ratio**: 11.59x

