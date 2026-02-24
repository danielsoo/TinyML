# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v24 |
| **Run (datetime)** | 2026-02-24_21-20-19 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-24 21:21:17 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.7 |
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
| **Ratio sweep models** | 5 models |
| **Always build traditional** | True |
| **Traditional model path** | null |

## Summary

Total stages analyzed: 5

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Baseline | 0.7797 | 204,880 | 0.1780 | 0.3016 | 0.0006 | 0.9750 | 1.89 |
| Traditional+PTQ | 0.0732 | 62,048 | 0.1787 | 0.3018 | 0.0014 | 0.9946 | 0.49 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9654 | 0.9055 | 0.9722 | 0.9855 | 0.48 |
| QAT+PTQ | 0.0731 | 62,048 | 0.9619 | 0.8958 | 0.9707 | 0.9829 | 0.48 |
| QAT+QAT | 0.0635 | 62,048 | 0.9602 | 0.8932 | 0.9648 | 0.9864 | 0.48 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -90.62% ↓ | +0.0661% ↑ | +0.0177% ↑ | -74.16% ↓ | ✅ All Improved |
| Traditional+QAT | -91.85% ↓ | +78.7383% ↑ | +60.3817% ↑ | -74.74% ↓ | ✅ All Improved |
| QAT+PTQ | -90.62% ↓ | +78.3940% ↑ | +59.4152% ↑ | -74.66% ↓ | ✅ All Improved |
| QAT+QAT | -91.85% ↓ | +78.2161% ↑ | +59.1562% ↑ | -74.51% ↓ | ✅ All Improved |

## Pipeline overview (How each stage is produced)

| Stage | Input | Processing | Output file |
|-------|-------|------------|-------------|
| **Keras** | FL/central training done | Use as-is (no QAT strip) or load .h5 | `models/global_model.h5` |
### 2×2 Experimental Design

| Model | Training Method | Compression Pipeline | Filename |
|-------|----------------|---------------------|----------|
| **Baseline** | QAT-trained | No compression (float32 TFLite) | `saved_model_original.tflite` |
| **Traditional + PTQ** | Traditional (no QAT) | 50% prune → **PTQ** → int8 TFLite | `saved_model_no_qat_ptq.tflite` |
| **Traditional + QAT** | Traditional (no QAT) | 50% prune → **QAT fine-tune (2 epochs)** → int8 TFLite | `saved_model_traditional_qat.tflite` |
| **QAT + PTQ** | QAT-trained | 50% prune → **PTQ** → int8 TFLite | `saved_model_qat_ptq.tflite` |
| **QAT + QAT** | QAT-trained | 50% prune → **QAT fine-tune (2 epochs)** → int8 TFLite | `saved_model_pruned_qat.tflite` |

## Compression Ratios

| Stage | Compression Ratio | Size Reduction |
|-------|------------------|----------------|
| Baseline | 1.00x | 0.0% |
| Traditional+PTQ | 10.66x | 90.6% |
| Traditional+QAT | 12.28x | 91.9% |
| QAT+PTQ | 10.66x | 90.6% |
| QAT+QAT | 12.28x | 91.9% |

## Detailed Metrics

### Baseline

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7797 MB (817,528 bytes)
- **Parameters**: 204,880
- **Accuracy**: 0.1780
- **Precision**: 0.1776
- **Recall**: 0.9999
- **F1-Score**: 0.3016
- **Normal Recall** (of actual normal, % predicted as normal): 0.0006
- **Normal Precision** (of predicted normal, % actually normal): 0.9750
- **Avg Latency**: 1.89 ms
- **Samples/sec**: 52876.93
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0732 MB (76,720 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.1787
- **Precision**: 0.1777
- **Recall**: 1.0000
- **F1-Score**: 0.3018
- **Normal Recall** (of actual normal, % predicted as normal): 0.0014
- **Normal Precision** (of predicted normal, % actually normal): 0.9946
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 204670.08
- **Compression Ratio**: 10.66x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9654
- **Precision**: 0.8787
- **Recall**: 0.9338
- **F1-Score**: 0.9055
- **Normal Recall** (of actual normal, % predicted as normal): 0.9722
- **Normal Precision** (of predicted normal, % actually normal): 0.9855
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 209327.94
- **Compression Ratio**: 12.28x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0731 MB (76,688 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9619
- **Precision**: 0.8714
- **Recall**: 0.9215
- **F1-Score**: 0.8958
- **Normal Recall** (of actual normal, % predicted as normal): 0.9707
- **Normal Precision** (of predicted normal, % actually normal): 0.9829
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208661.46
- **Compression Ratio**: 10.66x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9602
- **Precision**: 0.8521
- **Recall**: 0.9385
- **F1-Score**: 0.8932
- **Normal Recall** (of actual normal, % predicted as normal): 0.9648
- **Normal Precision** (of predicted normal, % actually normal): 0.9864
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 207443.69
- **Compression Ratio**: 12.28x

