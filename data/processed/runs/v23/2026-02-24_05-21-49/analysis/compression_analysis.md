# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v23 |
| **Run (datetime)** | 2026-02-24_05-21-49 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-24 05:22:48 |

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
| Baseline | 0.7797 | 204,880 | 0.1836 | 0.3019 | 0.0085 | 0.8761 | 1.87 |
| Traditional+PTQ | 0.0676 | 61,969 | 0.2204 | 0.3129 | 0.0521 | 1.0000 | 0.50 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9639 | 0.9020 | 0.9699 | 0.9860 | 0.49 |
| QAT+PTQ | 0.0654 | 61,520 | 0.1775 | 0.3015 | 0.0000 | 1.0000 | 0.47 |
| QAT+QAT | 0.0635 | 62,048 | 0.9578 | 0.8840 | 0.9693 | 0.9792 | 0.48 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -91.33% ↓ | +3.6838% ↑ | +1.1024% ↑ | -73.09% ↓ | ✅ All Improved |
| Traditional+QAT | -91.85% ↓ | +78.0315% ↑ | +60.0063% ↑ | -74.09% ↓ | ✅ All Improved |
| QAT+PTQ | -91.61% ↓ | -0.6020% ↓ | -0.0366% ↓ | -74.70% ↓ | ✅ Mostly Better |
| QAT+QAT | -91.85% ↓ | +77.4285% ↑ | +58.2085% ↑ | -74.34% ↓ | ✅ All Improved |

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
| Traditional+PTQ | 11.54x | 91.3% |
| Traditional+QAT | 12.28x | 91.9% |
| QAT+PTQ | 11.91x | 91.6% |
| QAT+QAT | 12.28x | 91.9% |

## Detailed Metrics

### Baseline

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7797 MB (817,528 bytes)
- **Parameters**: 204,880
- **Accuracy**: 0.1836
- **Precision**: 0.1780
- **Recall**: 0.9944
- **F1-Score**: 0.3019
- **Normal Recall** (of actual normal, % predicted as normal): 0.0085
- **Normal Precision** (of predicted normal, % actually normal): 0.8761
- **Avg Latency**: 1.87 ms
- **Samples/sec**: 53358.57
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
- **Avg Latency**: 0.50 ms
- **Samples/sec**: 198293.49
- **Compression Ratio**: 11.54x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9639
- **Precision**: 0.8703
- **Recall**: 0.9360
- **F1-Score**: 0.9020
- **Normal Recall** (of actual normal, % predicted as normal): 0.9699
- **Normal Precision** (of predicted normal, % actually normal): 0.9860
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 205905.94
- **Compression Ratio**: 12.28x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0654 MB (68,624 bytes)
- **Parameters**: 61,520
- **Accuracy**: 0.1775
- **Precision**: 0.1775
- **Recall**: 1.0000
- **F1-Score**: 0.3015
- **Normal Recall** (of actual normal, % predicted as normal): 0.0000
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 0.47 ms
- **Samples/sec**: 210864.41
- **Compression Ratio**: 11.91x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9578
- **Precision**: 0.8643
- **Recall**: 0.9046
- **F1-Score**: 0.8840
- **Normal Recall** (of actual normal, % predicted as normal): 0.9693
- **Normal Precision** (of predicted normal, % actually normal): 0.9792
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 207978.58
- **Compression Ratio**: 12.28x

