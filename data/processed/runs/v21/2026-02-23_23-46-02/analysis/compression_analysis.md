# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v21 |
| **Run (datetime)** | 2026-02-23_23-46-02 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-23 23:46:59 |

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
| **FL rounds** | 2 |
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

Total stages analyzed: 4

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Baseline | 0.7835 | 205,777 | 0.9155 | 0.7829 | 0.9279 | 0.9680 | 1.91 |
| Traditional+PTQ | 0.0676 | 61,969 | 0.2204 | 0.3129 | 0.0521 | 1.0000 | 0.48 |
| QAT+PTQ | 0.0676 | 61,969 | 0.9319 | 0.7825 | 0.9840 | 0.9364 | 0.48 |
| QAT+QAT | 0.0635 | 62,048 | 0.8906 | 0.6103 | 0.9786 | 0.8976 | 0.50 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -91.38% ↓ | -69.5135% ↓ | -47.0012% ↓ | -74.68% ↓ | ✅ Mostly Better |
| QAT+PTQ | -91.38% ↓ | +1.6339% ↑ | -0.0444% ↓ | -74.60% ↓ | ✅ Mostly Better |
| QAT+QAT | -91.89% ↓ | -2.4977% ↓ | -17.2587% ↓ | -73.96% ↓ | ✅ Mostly Better |

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
| Traditional+PTQ | 11.59x | 91.4% |
| QAT+PTQ | 11.60x | 91.4% |
| QAT+QAT | 12.34x | 91.9% |

## Detailed Metrics

### Baseline

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7835 MB (821,560 bytes)
- **Parameters**: 205,777
- **Accuracy**: 0.9155
- **Precision**: 0.7199
- **Recall**: 0.8581
- **F1-Score**: 0.7829
- **Normal Recall** (of actual normal, % predicted as normal): 0.9279
- **Normal Precision** (of predicted normal, % actually normal): 0.9680
- **Avg Latency**: 1.91 ms
- **Samples/sec**: 52461.59
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,856 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.2204
- **Precision**: 0.1855
- **Recall**: 1.0000
- **F1-Score**: 0.3129
- **Normal Recall** (of actual normal, % predicted as normal): 0.0521
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 207207.98
- **Compression Ratio**: 11.59x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0676 MB (70,848 bytes)
- **Parameters**: 61,969
- **Accuracy**: 0.9319
- **Precision**: 0.9032
- **Recall**: 0.6903
- **F1-Score**: 0.7825
- **Normal Recall** (of actual normal, % predicted as normal): 0.9840
- **Normal Precision** (of predicted normal, % actually normal): 0.9364
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 206524.40
- **Compression Ratio**: 11.60x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.8906
- **Precision**: 0.8294
- **Recall**: 0.4828
- **F1-Score**: 0.6103
- **Normal Recall** (of actual normal, % predicted as normal): 0.9786
- **Normal Precision** (of predicted normal, % actually normal): 0.8976
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 201445.85
- **Compression Ratio**: 12.34x

