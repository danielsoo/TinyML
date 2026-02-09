# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v13 |
| **Run (datetime)** | 2026-02-06_11-44-34 |
| **Data Version** | cicids2017_max1500k_bal4.0 |
| **Generated** | 2026-02-06 11:45:30 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 1500000 |
| **Balance ratio** (normal:attack) | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.92 |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 0.1 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |

## Summary

Total stages analyzed: 4

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.2370 | 62,036 | 0.0482 | 0.0061 | 0.0550 | 0.2076 | 0.77 |
| Compressed (PTQ) | 0.0254 | 20,820 | 0.1532 | 0.0837 | 0.1389 | 0.4549 | 0.40 |
| Original (no distill) | 0.7836 | 205,777 | 0.7746 | 0.5748 | 0.7551 | 0.9635 | 1.83 |
| Compressed PTQ (no distill) | 0.0676 | 61,969 | 0.5328 | 0.4182 | 0.4428 | 0.9786 | 0.50 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| Compressed (PTQ) | -89.28% ↓ | +10.4945% ↑ | +7.7637% ↑ | -47.73% ↓ | ✅ All Improved |
| Original (no distill) | +230.68% ↑ | +72.6376% ↑ | +56.8685% ↑ | +139.15% ↑ | ✅ Mostly Better |
| Compressed PTQ (no distill) | -71.46% ↓ | +48.4593% ↑ | +41.2103% ↑ | -35.31% ↓ | ✅ All Improved |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| Compressed (PTQ) | 9.32x | 89.3% |
| Original (no distill) | 0.30x | -230.7% |
| Compressed PTQ (no distill) | 3.50x | 71.5% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.2370 MB (248,488 bytes)
- **Parameters**: 62,036
- **Accuracy**: 0.0482
- **Precision**: 0.0037
- **Recall**: 0.0165
- **F1-Score**: 0.0061
- **Normal Recall** (of actual normal, % predicted as normal): 0.0550
- **Normal Precision** (of predicted normal, % actually normal): 0.2076
- **Avg Latency**: 0.77 ms
- **Samples/sec**: 130594.51
- **Compression Ratio**: 1.00x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0254 MB (26,648 bytes)
- **Parameters**: 20,820
- **Accuracy**: 0.1532
- **Precision**: 0.0517
- **Recall**: 0.2199
- **F1-Score**: 0.0837
- **Normal Recall** (of actual normal, % predicted as normal): 0.1389
- **Normal Precision** (of predicted normal, % actually normal): 0.4549
- **Avg Latency**: 0.40 ms
- **Samples/sec**: 249824.53
- **Compression Ratio**: 9.32x

### Original (no distill)

- **Model Path**: `models/tflite/saved_model_original_no_distill.tflite`
- **File Size**: 0.7836 MB (821,688 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.7746
- **Precision**: 0.4301
- **Recall**: 0.8661
- **F1-Score**: 0.5748
- **Normal Recall** (of actual normal, % predicted as normal): 0.7551
- **Normal Precision** (of predicted normal, % actually normal): 0.9635
- **Avg Latency**: 1.83 ms
- **Samples/sec**: 54606.93
- **Compression Ratio**: 0.30x

### Compressed PTQ (no distill)

- **Model Path**: `models/tflite/saved_model_pruned_quantized_no_distill.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5328
- **Precision**: 0.2677
- **Recall**: 0.9546
- **F1-Score**: 0.4182
- **Normal Recall** (of actual normal, % predicted as normal): 0.4428
- **Normal Precision** (of predicted normal, % actually normal): 0.9786
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 201882.17
- **Compression Ratio**: 3.50x

