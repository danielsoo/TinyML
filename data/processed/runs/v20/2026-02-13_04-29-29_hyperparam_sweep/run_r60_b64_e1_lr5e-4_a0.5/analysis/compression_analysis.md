# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-13_08-43-42 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-13 08:44:55 |

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
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.5 |
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
| Original | 0.7836 | 205,777 | 0.3244 | 0.2558 | 0.2533 | 0.7723 | 2.24 |
| QAT+Prune only | 0.2367 | 61,969 | 0.7468 | 0.5828 | 0.6929 | 0.9989 | 0.90 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.7480 | 0.5840 | 0.6943 | 0.9989 | 0.54 |
| QAT+PTQ | 0.0676 | 61,969 | 0.7480 | 0.5840 | 0.6943 | 0.9989 | 0.54 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.54 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +42.2352% ↑ | +32.7045% ↑ | -59.70% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +42.3558% ↑ | +32.8210% ↑ | -75.80% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +42.3558% ↑ | +32.8210% ↑ | -75.91% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +49.8053% ↑ | -25.5779% ↓ | -75.84% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.3244
- **Precision**: 0.1590
- **Recall**: 0.6539
- **F1-Score**: 0.2558
- **Normal Recall** (of actual normal, % predicted as normal): 0.2533
- **Normal Precision** (of predicted normal, % actually normal): 0.7723
- **Avg Latency**: 2.24 ms
- **Samples/sec**: 44560.05
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7468
- **Precision**: 0.4119
- **Recall**: 0.9964
- **F1-Score**: 0.5828
- **Normal Recall** (of actual normal, % predicted as normal): 0.6929
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.90 ms
- **Samples/sec**: 110562.63
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7480
- **Precision**: 0.4130
- **Recall**: 0.9964
- **F1-Score**: 0.5840
- **Normal Recall** (of actual normal, % predicted as normal): 0.6943
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 184138.38
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7480
- **Precision**: 0.4130
- **Recall**: 0.9964
- **F1-Score**: 0.5840
- **Normal Recall** (of actual normal, % predicted as normal): 0.6943
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 184950.35
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
- **Avg Latency**: 0.54 ms
- **Samples/sec**: 184421.76
- **Compression Ratio**: 11.59x

