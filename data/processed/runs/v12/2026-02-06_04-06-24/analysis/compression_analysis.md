# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-02-06_04-06-24 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-06 04:07:17 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 1500000 |
| **Balance ratio** (정상:공격) | 4.0 (정상:공격 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 50 |
| **Local epochs** | 5 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.92 |
| **Use QAT** | True |
| **Server momentum** | 0.9 |
| **Server learning rate** | 1.0 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 2

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.6467 | 0.0304 | 0.7780 | 0.7901 | 1.88 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.5863 | 0.4475 | 0.5081 | 0.9805 | 0.50 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -91.37% ↓ | -6.0442% ↓ | +41.7069% ↑ | -73.20% ↓ | ✅ Mostly Better |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,652 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.6467
- **Precision**: 0.0294
- **Recall**: 0.0315
- **F1-Score**: 0.0304
- **Normal Recall** (of actual normal, % predicted as normal): 0.7780
- **Normal Precision** (of predicted normal, % actually normal): 0.7901
- **Avg Latency**: 1.88 ms
- **Samples/sec**: 53155.03
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,896 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5863
- **Precision**: 0.2924
- **Recall**: 0.9525
- **F1-Score**: 0.4475
- **Normal Recall** (of actual normal, % predicted as normal): 0.5081
- **Normal Precision** (of predicted normal, % actually normal): 0.9805
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 198359.14
- **Compression Ratio**: 11.59x

