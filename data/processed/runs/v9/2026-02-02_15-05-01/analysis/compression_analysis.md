# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v9 |
| **Run (datetime)** | 2026-02-02_15-05-01 |
| **Data Version** | cicids2017_max1500k_bal10.0 |
| **Generated** | 2026-02-02 15:05:59 |## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 1500000 |
| **Balance ratio** (정상:공격) | 4.0   # 공격 2 : 정상 8 (실제 트래픽 5~20% 공격 가정) |
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
| **Focal loss alpha** | 0.75 |
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
| Original | 0.7839 | 205,777 | 0.7853 | 0.0000 | 1.22 |
| Compressed | 0.0731 | 61,969 | 0.7853 | 0.0013 | 0.71 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -90.68% ↓ | -0.0059% ↓ | +0.1308% ↑ | -41.67% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.7853
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Avg Latency**: 1.22 ms
- **Samples/sec**: 82205.79
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0731 MB (76,640 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7853
- **Precision**: 0.4130
- **Recall**: 0.0007
- **F1-Score**: 0.0013
- **Avg Latency**: 0.71 ms
- **Samples/sec**: 140923.43
- **Compression Ratio**: 10.73x

