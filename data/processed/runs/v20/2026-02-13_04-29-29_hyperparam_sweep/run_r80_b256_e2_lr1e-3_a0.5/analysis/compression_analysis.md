# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-19_03-02-25 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-19 03:03:44 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
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
| Original | 0.7836 | 205,777 | 0.9021 | 0.7599 | 0.9084 | 0.9707 | 2.26 |
| QAT+Prune only | 0.2367 | 61,969 | 0.5034 | 0.4164 | 0.3967 | 0.9989 | 0.87 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.5041 | 0.4167 | 0.3975 | 0.9989 | 0.55 |
| QAT+PTQ | 0.0676 | 61,969 | 0.5041 | 0.4167 | 0.3975 | 0.9989 | 0.55 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.54 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | -39.8630% ↓ | -34.3464% ↓ | -61.67% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.37% ↓ | -39.7933% ↓ | -34.3127% ↓ | -75.82% ↓ | ✅ Mostly Better |
| QAT+PTQ | -91.37% ↓ | -39.7933% ↓ | -34.3127% ↓ | -75.77% ↓ | ✅ Mostly Better |
| noQAT+PTQ | -91.37% ↓ | -7.9592% ↓ | -75.9875% ↓ | -76.19% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.9021
- **Precision**: 0.6728
- **Recall**: 0.8729
- **F1-Score**: 0.7599
- **Normal Recall** (of actual normal, % predicted as normal): 0.9084
- **Normal Precision** (of predicted normal, % actually normal): 0.9707
- **Avg Latency**: 2.26 ms
- **Samples/sec**: 44206.87
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5034
- **Precision**: 0.2631
- **Recall**: 0.9979
- **F1-Score**: 0.4164
- **Normal Recall** (of actual normal, % predicted as normal): 0.3967
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.87 ms
- **Samples/sec**: 115323.18
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5041
- **Precision**: 0.2634
- **Recall**: 0.9979
- **F1-Score**: 0.4167
- **Normal Recall** (of actual normal, % predicted as normal): 0.3975
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 182861.93
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.5041
- **Precision**: 0.2634
- **Recall**: 0.9979
- **F1-Score**: 0.4167
- **Normal Recall** (of actual normal, % predicted as normal): 0.3975
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 182472.11
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
- **Samples/sec**: 185654.39
- **Compression Ratio**: 11.59x

