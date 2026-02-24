# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-16_15-32-17 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-16 15:33:22 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/CIC-IDS2017 |
| **Max samples** | 2000000 |
| **Balance ratio** (normal:attack) | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.65 |
| **Use distillation** | False |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 1.0 |
| **LR decay type** | cosine |
| **LR decay rate** | 0.97 |
| **LR drop rate** | 0.5 |
| **LR epochs drop** | 5 |
| **LR min** | 0.0001 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |
| **Min available clients** | 4 |
| **Prediction threshold** | 0.3 |
| **Ratio sweep models** | 6 models |
| **Always build traditional** | True |
| **Traditional model path** | null |

## Summary

Total stages analyzed: 5

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Original | 0.7836 | 205,777 | 0.6632 | 0.3986 | 0.6707 | 0.8932 | 1.91 |
| QAT+Prune only | 0.2367 | 61,969 | 0.8676 | 0.7265 | 0.8410 | 0.9976 | 0.84 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.8671 | 0.7258 | 0.8405 | 0.9976 | 0.51 |
| QAT+PTQ | 0.0676 | 61,969 | 0.8671 | 0.7258 | 0.8405 | 0.9976 | 0.49 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.50 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +20.4324% ↑ | +32.7881% ↑ | -56.33% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +20.3906% ↑ | +32.7210% ↑ | -73.14% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +20.3906% ↑ | +32.7210% ↑ | -74.29% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +15.9239% ↑ | -39.8607% ↓ | -73.61% ↓ | ✅ Mostly Better |

## Pipeline overview (How each stage is produced)

| Stage | Input | Processing | Output file |
|-------|-------|------------|-------------|
| **Keras** | FL/central training done | Use as-is (no QAT strip) or load .h5 | `models/global_model.h5` |
| **Original (TFLite)** | Keras model | float32 TFLite export (no quantization) | `saved_model_original.tflite` |
| **QAT+Prune only** | QAT-trained model | QAT strip → 50% structured pruning → **float32** TFLite (no quant) | `saved_model_qat_pruned_float32.tflite` |
| **Compressed (QAT)** | QAT-trained model | QAT strip → 50% prune → **quantize_model** → 2 epoch fine-tune → **int8** TFLite | `saved_model_pruned_qat.tflite` |
| **QAT+PTQ** | QAT-trained model | QAT strip → 50% prune → **PTQ** (quantize with representative data) → int8 TFLite | `saved_model_qat_ptq.tflite` |
| **noQAT+PTQ** | Model trained **without QAT** | 50% prune → PTQ → int8 TFLite | `saved_model_no_qat_ptq.tflite` |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Original | 1.00x | 0.0% |
| QAT+Prune only | 3.31x | 69.8% |
| Compressed (PTQ) | 11.59x | 91.4% |
| QAT+PTQ | 11.59x | 91.4% |
| noQAT+PTQ | 11.59x | 91.4% |

## Detailed Metrics

### Original

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7836 MB (821,632 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.6632
- **Precision**: 0.2918
- **Recall**: 0.6287
- **F1-Score**: 0.3986
- **Normal Recall** (of actual normal, % predicted as normal): 0.6707
- **Normal Precision** (of predicted normal, % actually normal): 0.8932
- **Avg Latency**: 1.91 ms
- **Samples/sec**: 52265.47
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8676
- **Precision**: 0.5735
- **Recall**: 0.9908
- **F1-Score**: 0.7265
- **Normal Recall** (of actual normal, % predicted as normal): 0.8410
- **Normal Precision** (of predicted normal, % actually normal): 0.9976
- **Avg Latency**: 0.84 ms
- **Samples/sec**: 119669.72
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8671
- **Precision**: 0.5727
- **Recall**: 0.9906
- **F1-Score**: 0.7258
- **Normal Recall** (of actual normal, % predicted as normal): 0.8405
- **Normal Precision** (of predicted normal, % actually normal): 0.9976
- **Avg Latency**: 0.51 ms
- **Samples/sec**: 194586.13
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8671
- **Precision**: 0.5727
- **Recall**: 0.9906
- **F1-Score**: 0.7258
- **Normal Recall** (of actual normal, % predicted as normal): 0.8405
- **Normal Precision** (of predicted normal, % actually normal): 0.9976
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 203251.79
- **Compression Ratio**: 11.59x

### noQAT+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.8225
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, % predicted as normal): 1.0000
- **Normal Precision** (of predicted normal, % actually normal): 0.8225
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 198087.47
- **Compression Ratio**: 11.59x

