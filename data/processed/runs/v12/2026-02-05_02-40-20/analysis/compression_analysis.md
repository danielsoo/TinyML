# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-05_02-40-20 |
| **Data Version** | cicids2017_bal1.0 |
| **Generated** | 2026-02-05 02:41:11 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | /scratch/yqp5187/Bot-IoT |
| **Max samples** | - |
| **Balance ratio** (정상:공격) | 1.0 (50:50) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **Aggregation strategy** | FedAvgM (momentum=0.9) |
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
| Original | 0.7836 | 205,777 | 0.8307 | 0.2351 | 0.9700 | 0.8479 | 1.41 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9120 | 0.6902 | 0.9811 | 0.9184 | 0.40 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | +8.1237% ↑ | +45.5097% ↑ | -71.59% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.8307
- **Precision**: 0.5110
- **Recall**: 0.1527
- **F1-Score**: 0.2351
- **Normal Recall** (of actual normal, % predicted as normal): 0.9700
- **Normal Precision** (of predicted normal, % actually normal): 0.8479
- **Avg Latency**: 1.41 ms
- **Samples/sec**: 70819.82
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9120
- **Precision**: 0.8620
- **Recall**: 0.5755
- **F1-Score**: 0.6902
- **Normal Recall** (of actual normal, % predicted as normal): 0.9811
- **Normal Precision** (of predicted normal, % actually normal): 0.9184
- **Avg Latency**: 0.40 ms
- **Samples/sec**: 249260.36
- **Compression Ratio**: 11.59x

