# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-13_03-40-59 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-13 03:42:14 |

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
| **Learning rate** | 0.001 |
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
| Original | 0.7836 | 205,777 | 0.5506 | 0.4022 | 0.4857 | 0.9381 | 2.05 |
| QAT+Prune only | 0.2367 | 61,969 | 0.7628 | 0.5988 | 0.7122 | 0.9992 | 0.72 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.7663 | 0.6024 | 0.7164 | 0.9992 | 0.50 |
| QAT+PTQ | 0.0676 | 61,969 | 0.7663 | 0.6024 | 0.7164 | 0.9992 | 0.48 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.52 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +21.2151% ↑ | +19.6594% ↑ | -64.94% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +21.5655% ↑ | +20.0151% ↑ | -75.52% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +21.5655% ↑ | +20.0151% ↑ | -76.58% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +27.1839% ↑ | -40.2224% ↓ | -74.85% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.5506
- **Precision**: 0.2633
- **Recall**: 0.8516
- **F1-Score**: 0.4022
- **Normal Recall** (of actual normal, % predicted as normal): 0.4857
- **Normal Precision** (of predicted normal, % actually normal): 0.9381
- **Avg Latency**: 2.05 ms
- **Samples/sec**: 48739.81
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,212 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7628
- **Precision**: 0.4279
- **Recall**: 0.9972
- **F1-Score**: 0.5988
- **Normal Recall** (of actual normal, % predicted as normal): 0.7122
- **Normal Precision** (of predicted normal, % actually normal): 0.9992
- **Avg Latency**: 0.72 ms
- **Samples/sec**: 139022.34
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7663
- **Precision**: 0.4315
- **Recall**: 0.9972
- **F1-Score**: 0.6024
- **Normal Recall** (of actual normal, % predicted as normal): 0.7164
- **Normal Precision** (of predicted normal, % actually normal): 0.9992
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 199112.46
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.7663
- **Precision**: 0.4315
- **Recall**: 0.9972
- **F1-Score**: 0.6024
- **Normal Recall** (of actual normal, % predicted as normal): 0.7164
- **Normal Precision** (of predicted normal, % actually normal): 0.9992
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208154.04
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
- **Avg Latency**: 0.52 ms
- **Samples/sec**: 193812.86
- **Compression Ratio**: 11.59x

