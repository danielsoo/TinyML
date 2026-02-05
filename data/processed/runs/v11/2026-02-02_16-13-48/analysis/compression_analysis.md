# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v11 |
| **Run (datetime)** | 2026-02-02_16-13-48 |
| **Data Version** | cicids2017_max1500k_bal1.0 |
| **Generated** | 2026-02-02 16:14:14 |## Run / Training Configuration

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
| Original | 0.7839 | 205,777 | 0.8242 | 0.0000 | 1.73 |
| Compressed | 0.0731 | 61,969 | 0.8386 | 0.1514 | 0.70 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -90.68% ↓ | +1.4335% ↑ | +15.1367% ↑ | -59.62% ↓ | ✅ All Improved |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed | 10.73x | 90.7% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7839 MB (822,012 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.8242
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 1.73 ms
- **Samples/sec**: 57878.02
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0731 MB (76,640 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8386
- **Precision**: 0.9956
- **Recall**: 0.0819
- **F1-Score**: 0.1514
- **Avg Latency**: 0.70 ms
- **Samples/sec**: 143321.51
- **Compression Ratio**: 10.73x

