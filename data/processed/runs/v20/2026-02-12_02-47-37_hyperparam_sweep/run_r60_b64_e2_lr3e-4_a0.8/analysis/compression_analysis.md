# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-12_19-26-44 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-12 19:28:01 |

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
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.8 |
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
| Original | 0.7836 | 205,777 | 0.6970 | 0.3287 | 0.7573 | 0.8577 | 2.16 |
| QAT+Prune only | 0.2367 | 61,969 | 0.9268 | 0.7946 | 0.9546 | 0.9563 | 0.84 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.9259 | 0.7931 | 0.9531 | 0.9567 | 0.55 |
| QAT+PTQ | 0.0676 | 61,969 | 0.9259 | 0.7931 | 0.9531 | 0.9567 | 0.57 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.54 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +22.9758% ↑ | +46.5981% ↑ | -61.19% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +22.8871% ↑ | +46.4447% ↑ | -74.50% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +22.8871% ↑ | +46.4447% ↑ | -73.66% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +12.5445% ↑ | -32.8653% ↓ | -75.01% ↓ | ✅ Mostly Better |

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
- **File Size**: 0.7836 MB (821,652 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.6970
- **Precision**: 0.2709
- **Recall**: 0.4177
- **F1-Score**: 0.3287
- **Normal Recall** (of actual normal, % predicted as normal): 0.7573
- **Normal Precision** (of predicted normal, % actually normal): 0.8577
- **Avg Latency**: 2.16 ms
- **Samples/sec**: 46220.26
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,212 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9268
- **Precision**: 0.7914
- **Recall**: 0.7979
- **F1-Score**: 0.7946
- **Normal Recall** (of actual normal, % predicted as normal): 0.9546
- **Normal Precision** (of predicted normal, % actually normal): 0.9563
- **Avg Latency**: 0.84 ms
- **Samples/sec**: 119102.23
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9259
- **Precision**: 0.7863
- **Recall**: 0.8000
- **F1-Score**: 0.7931
- **Normal Recall** (of actual normal, % predicted as normal): 0.9531
- **Normal Precision** (of predicted normal, % actually normal): 0.9567
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 181234.24
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9259
- **Precision**: 0.7863
- **Recall**: 0.8000
- **F1-Score**: 0.7931
- **Normal Recall** (of actual normal, % predicted as normal): 0.9531
- **Normal Precision** (of predicted normal, % actually normal): 0.9567
- **Avg Latency**: 0.57 ms
- **Samples/sec**: 175449.85
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
- **Samples/sec**: 184958.50
- **Compression Ratio**: 11.59x

