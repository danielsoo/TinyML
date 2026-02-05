# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-02_23-28-45 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-02 23:29:11 |## Run / Training Configuration

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

## Comparison Table

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |
|-------|-----------|------------|----------|----------|--------------|
| Original | 0.7836 | 205,777 | 0.9373 | 0.8461 | 1.54 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9020 | 0.6263 | 0.43 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -3.5288% ↓ | -21.9792% ↓ | -71.79% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.9373
- **Precision**: 0.7440
- **Recall**: 0.9807
- **F1-Score**: 0.8461
- **Avg Latency**: 1.54 ms
- **Samples/sec**: 65093.57
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9020
- **Precision**: 0.9498
- **Recall**: 0.4672
- **F1-Score**: 0.6263
- **Avg Latency**: 0.43 ms
- **Samples/sec**: 230735.17
- **Compression Ratio**: 11.59x

