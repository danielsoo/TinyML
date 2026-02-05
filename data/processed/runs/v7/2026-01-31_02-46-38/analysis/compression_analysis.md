# Compression Analysis Report

| 항목 | 값 |
|------|----|
| **Version** | v7 |
| **Run (날짜_시간)** | 2026-01-31_02-46-38 |
| **Data Version** | cicids2017_max1500k |
| **Generated** | 2026-01-31 02:47:23 |## Run / Training Configuration

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
| Original | 0.2056 | 53,844 | 0.1293 | 0.2207 | 0.46 |
| Compressed | 0.0206 | 18,772 | 0.2147 | 0.3534 | 0.68 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed | -89.96% ↓ | +8.5330% ↑ | +13.2699% ↑ | +47.32% ↑ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed | 9.96x | 90.0% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.2056 MB (215,612 bytes)
- **Parameters**: 53,844
- **Accuracy**: 0.1293
- **Precision**: 0.1366
- **Recall**: 0.5745
- **F1-Score**: 0.2207
- **Avg Latency**: 0.46 ms
- **Samples/sec**: 216323.89
- **Compression Ratio**: 1.00x

### Compressed

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0206 MB (21,648 bytes)
- **Parameters**: 18,772
- **Accuracy**: 0.2147
- **Precision**: 0.2147
- **Recall**: 1.0000
- **F1-Score**: 0.3534
- **Avg Latency**: 0.68 ms
- **Samples/sec**: 146843.96
- **Compression Ratio**: 9.96x

