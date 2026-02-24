# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-12_04-11-09 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-12 04:12:27 |

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
| **Learning rate** | 0.0003 |
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
| Original | 0.7836 | 205,777 | 0.6646 | 0.4279 | 0.6556 | 0.9118 | 2.10 |
| QAT+Prune only | 0.2367 | 61,969 | 0.6697 | 0.5172 | 0.5990 | 0.9989 | 0.83 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.6704 | 0.5178 | 0.6000 | 0.9988 | 0.48 |
| QAT+PTQ | 0.0676 | 61,969 | 0.6704 | 0.5178 | 0.6000 | 0.9988 | 0.53 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.51 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | +0.5012% ↑ | +8.9367% ↑ | -60.43% ↓ | ✅ All Improved |
| Compressed (PTQ) | -91.37% ↓ | +0.5770% ↑ | +8.9932% ↑ | -77.16% ↓ | ✅ All Improved |
| QAT+PTQ | -91.37% ↓ | +0.5770% ↑ | +8.9932% ↑ | -74.67% ↓ | ✅ All Improved |
| noQAT+PTQ | -91.37% ↓ | +15.7828% ↑ | -42.7868% ↓ | -75.81% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.6646
- **Precision**: 0.3069
- **Recall**: 0.7063
- **F1-Score**: 0.4279
- **Normal Recall** (of actual normal, % predicted as normal): 0.6556
- **Normal Precision** (of predicted normal, % actually normal): 0.9118
- **Avg Latency**: 2.10 ms
- **Samples/sec**: 47651.17
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,212 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.6697
- **Precision**: 0.3492
- **Recall**: 0.9968
- **F1-Score**: 0.5172
- **Normal Recall** (of actual normal, % predicted as normal): 0.5990
- **Normal Precision** (of predicted normal, % actually normal): 0.9989
- **Avg Latency**: 0.83 ms
- **Samples/sec**: 120422.16
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.6704
- **Precision**: 0.3497
- **Recall**: 0.9968
- **F1-Score**: 0.5178
- **Normal Recall** (of actual normal, % predicted as normal): 0.6000
- **Normal Precision** (of predicted normal, % actually normal): 0.9988
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208619.95
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.6704
- **Precision**: 0.3497
- **Recall**: 0.9968
- **F1-Score**: 0.5178
- **Normal Recall** (of actual normal, % predicted as normal): 0.6000
- **Normal Precision** (of predicted normal, % actually normal): 0.9988
- **Avg Latency**: 0.53 ms
- **Samples/sec**: 188102.25
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
- **Avg Latency**: 0.51 ms
- **Samples/sec**: 197008.17
- **Compression Ratio**: 11.59x

