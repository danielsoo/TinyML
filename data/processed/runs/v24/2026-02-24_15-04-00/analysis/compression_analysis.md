# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v24 |
| **Run (datetime)** | 2026-02-24_15-04-00 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-24 15:04:58 |

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
| **FL rounds** | 1 |
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
| Baseline | 0.7797 | 204,880 | 0.1779 | 0.3016 | 0.0004 | 1.0000 | 1.91 |
| Traditional+PTQ | 0.0732 | 62,048 | 0.1775 | 0.3015 | 0.0000 | 0.0000 | 0.49 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9531 | 0.8788 | 0.9519 | 0.9907 | 0.49 |
| QAT+PTQ | 0.0731 | 62,048 | 0.9524 | 0.8740 | 0.9573 | 0.9844 | 0.48 |
| QAT+QAT | 0.0635 | 62,048 | 0.9643 | 0.9020 | 0.9726 | 0.9838 | 0.48 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -90.62% ↓ | -0.0356% ↓ | -0.0091% ↓ | -74.28% ↓ | ✅ Mostly Better |
| Traditional+QAT | -91.85% ↓ | +77.5178% ↑ | +57.7192% ↑ | -74.12% ↓ | ✅ All Improved |
| QAT+PTQ | -90.62% ↓ | +77.4511% ↑ | +57.2341% ↑ | -74.87% ↓ | ✅ All Improved |
| QAT+QAT | -91.85% ↓ | +78.6396% ↑ | +60.0360% ↑ | -74.85% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.1779
- **Precision**: 0.1776
- **Recall**: 1.0000
- **F1-Score**: 0.3016
- **Normal Recall** (of actual normal, % predicted as normal): 0.0004
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 1.91 ms
- **Samples/sec**: 52336.56
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0732 MB (76,720 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.1775
- **Precision**: 0.1775
- **Recall**: 1.0000
- **F1-Score**: 0.3015
- **Normal Recall** (of actual normal, % predicted as normal): 0.0000
- **Normal Precision** (of predicted normal, % actually normal): 0.0000
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 203458.84
- **Compression Ratio**: 10.66x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9531
- **Precision**: 0.8113
- **Recall**: 0.9586
- **F1-Score**: 0.8788
- **Normal Recall** (of actual normal, % predicted as normal): 0.9519
- **Normal Precision** (of predicted normal, % actually normal): 0.9907
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 202252.10
- **Compression Ratio**: 12.28x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0731 MB (76,688 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9524
- **Precision**: 0.8246
- **Recall**: 0.9296
- **F1-Score**: 0.8740
- **Normal Recall** (of actual normal, % predicted as normal): 0.9573
- **Normal Precision** (of predicted normal, % actually normal): 0.9844
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208247.06
- **Compression Ratio**: 10.66x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9643
- **Precision**: 0.8795
- **Recall**: 0.9257
- **F1-Score**: 0.9020
- **Normal Recall** (of actual normal, % predicted as normal): 0.9726
- **Normal Precision** (of predicted normal, % actually normal): 0.9838
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208102.41
- **Compression Ratio**: 12.28x

