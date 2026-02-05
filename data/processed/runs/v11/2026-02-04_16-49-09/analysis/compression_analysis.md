# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-04_16-49-09 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-04 16:49:47 |## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | /scratch/yqp5187/Bot-IoT |
| **Max samples** | 1500000 |
| **Balance ratio** (정상:공격) | 1.0 (50:50) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 25 |
| **Local epochs** | 15 |
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
| Original | 0.7836 | 205,777 | 0.8897 | 0.5783 | 0.9876 | 0.8905 | 1.50 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.7955 | 0.5759 | 0.7967 | 0.9468 | 0.42 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -9.4170% ↓ | -0.2371% ↓ | -71.67% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.8897
- **Precision**: 0.8812
- **Recall**: 0.4303
- **F1-Score**: 0.5783
- **Normal Recall** (of actual normal, % predicted as normal): 0.9876
- **Normal Precision** (of predicted normal, % actually normal): 0.8905
- **Avg Latency**: 1.50 ms
- **Samples/sec**: 66860.68
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7955
- **Precision**: 0.4531
- **Recall**: 0.7899
- **F1-Score**: 0.5759
- **Normal Recall** (of actual normal, % predicted as normal): 0.7967
- **Normal Precision** (of predicted normal, % actually normal): 0.9468
- **Avg Latency**: 0.42 ms
- **Samples/sec**: 236046.15
- **Compression Ratio**: 11.59x

