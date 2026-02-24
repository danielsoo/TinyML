# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v20 |
| **Run (datetime)** | 2026-02-14_11-54-38 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-14 11:56:00 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
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
| Original | 0.7836 | 205,777 | 0.5190 | 0.4227 | 0.4168 | 0.9959 | 2.21 |
| QAT+Prune only | 0.2367 | 61,969 | 0.3640 | 0.3579 | 0.2270 | 0.9986 | 0.87 |
| Compressed (PTQ) | 0.0676 | 61,969 | 0.3634 | 0.3577 | 0.2263 | 0.9986 | 0.53 |
| QAT+PTQ | 0.0676 | 61,969 | 0.3634 | 0.3577 | 0.2263 | 0.9986 | 0.55 |
| noQAT+PTQ | 0.0676 | 61,969 | 0.8225 | 0.0000 | 1.0000 | 0.8225 | 0.53 |

## 📊 Improvements vs Baseline

**Baseline:** Original

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Original | - | - | - | - | **Baseline** |
| QAT+Prune only | -69.79% ↓ | -15.4992% ↓ | -6.4811% ↓ | -60.75% ↓ | ✅ Mostly Better |
| Compressed (PTQ) | -91.37% ↓ | -15.5583% ↓ | -6.5028% ↓ | -75.92% ↓ | ✅ Mostly Better |
| QAT+PTQ | -91.37% ↓ | -15.5583% ↓ | -6.5028% ↓ | -75.25% ↓ | ✅ Mostly Better |
| noQAT+PTQ | -91.37% ↓ | +30.3506% ↑ | -42.2721% ↓ | -75.84% ↓ | ✅ Mostly Better |

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
- **Accuracy**: 0.5190
- **Precision**: 0.2686
- **Recall**: 0.9921
- **F1-Score**: 0.4227
- **Normal Recall** (of actual normal, % predicted as normal): 0.4168
- **Normal Precision** (of predicted normal, % actually normal): 0.9959
- **Avg Latency**: 2.21 ms
- **Samples/sec**: 45334.03
- **Compression Ratio**: 1.00x

### QAT+Prune only

- **Model Path**: `models/tflite/saved_model_qat_pruned_float32.tflite`
- **File Size**: 0.2367 MB (248,228 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.3640
- **Precision**: 0.2180
- **Recall**: 0.9985
- **F1-Score**: 0.3579
- **Normal Recall** (of actual normal, % predicted as normal): 0.2270
- **Normal Precision** (of predicted normal, % actually normal): 0.9986
- **Avg Latency**: 0.87 ms
- **Samples/sec**: 115501.02
- **Compression Ratio**: 3.31x

### Compressed (PTQ)

- **Model Path**: `models/tflite/saved_model_pruned_quantized.tflite`
- **File Size**: 0.0676 MB (70,920 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.3634
- **Precision**: 0.2179
- **Recall**: 0.9985
- **F1-Score**: 0.3577
- **Normal Recall** (of actual normal, % predicted as normal): 0.2263
- **Normal Precision** (of predicted normal, % actually normal): 0.9986
- **Avg Latency**: 0.53 ms
- **Samples/sec**: 188271.12
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,904 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.3634
- **Precision**: 0.2179
- **Recall**: 0.9985
- **F1-Score**: 0.3577
- **Normal Recall** (of actual normal, % predicted as normal): 0.2263
- **Normal Precision** (of predicted normal, % actually normal): 0.9986
- **Avg Latency**: 0.55 ms
- **Samples/sec**: 183181.38
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
- **Avg Latency**: 0.53 ms
- **Samples/sec**: 187639.42
- **Compression Ratio**: 11.59x

