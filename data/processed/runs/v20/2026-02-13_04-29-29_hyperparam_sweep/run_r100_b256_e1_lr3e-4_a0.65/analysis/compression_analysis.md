# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-22_02-58-16 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-22 02:59:16 |

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
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
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
| Original | 0.7836 | 205,777 | 0.5081 | 0.4168 | 0.4041 | 0.9948 | 1.91 |
| QAT+Prune only | 0.2367 | 61,969 | 0.5241 | 0.4254 | 0.4231 | 0.9961 | 0.78 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.5223 | 0.4245 | 0.4208 | 0.9961 | 0.48 |
| QAT+PTQ | 0.0676 | 61,969 | 0.5223 | 0.4245 | 0.4208 | 0.9961 | 0.47 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.47 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +1.5992% ↑ | +0.8602% ↑ | -58.97% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +1.4128% ↑ | +0.7647% ↑ | -74.99% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +1.4128% ↑ | +0.7647% ↑ | -75.20% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +31.4322% ↑ | -41.6842% ↓ | -75.26% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.5081
- **Precision**: 0.2640
- **Recall**: 0.9902
- **F1-Score**: 0.4168
- **Normal Recall** (of actual normal, % predicted as normal): 0.4041
- **Normal Precision** (of predicted normal, % actually normal): 0.9948
- **Avg Latency**: 1.91 ms
- **Samples/sec**: 52472.09
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5241
- **Precision**: 0.2708
- **Recall**: 0.9924
- **F1-Score**: 0.4254
- **Normal Recall** (of actual normal, % predicted as normal): 0.4231
- **Normal Precision** (of predicted normal, % actually normal): 0.9961
- **Avg Latency**: 0.78 ms
- **Samples/sec**: 127898.52
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5223
- **Precision**: 0.2700
- **Recall**: 0.9924
- **F1-Score**: 0.4245
- **Normal Recall** (of actual normal, % predicted as normal): 0.4208
- **Normal Precision** (of predicted normal, % actually normal): 0.9961
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 209809.61
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5223
- **Precision**: 0.2700
- **Recall**: 0.9924
- **F1-Score**: 0.4245
- **Normal Recall** (of actual normal, % predicted as normal): 0.4208
- **Normal Precision** (of predicted normal, % actually normal): 0.9961
- **Avg Latency**: 0.47 ms
- **Samples/sec**: 211545.06
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
- **Avg Latency**: 0.47 ms
- **Samples/sec**: 212112.07
- **Compression Ratio**: 11.59x

