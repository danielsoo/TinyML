# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v23 |
| **Run (datetime)** | 2026-02-24_13-27-12 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-24 13:28:09 |

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
| **Ratio sweep models** | 5 models |
| **Always build traditional** | True |
| **Traditional model path** | null |

## Summary

Total stages analyzed: 5

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Baseline | 0.7797 | 204,880 | 0.1778 | 0.3010 | 0.0010 | 0.6093 | 1.91 |
| Traditional+PTQ | 0.0676 | 61,969 | 0.2204 | 0.3129 | 0.0521 | 1.0000 | 0.49 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9638 | 0.9020 | 0.9691 | 0.9866 | 0.48 |
| QAT+PTQ | 0.0654 | 61,520 | 0.1775 | 0.3015 | 0.0000 | 0.0000 | 0.48 |
| QAT+QAT | 0.0635 | 62,048 | 0.9576 | 0.8854 | 0.9653 | 0.9829 | 0.49 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -91.33% ↓ | +4.2581% ↑ | +1.1926% ↑ | -74.18% ↓ | ✅ All Improved |
| Traditional+QAT | -91.85% ↓ | +78.5942% ↑ | +60.0964% ↑ | -75.00% ↓ | ✅ All Improved |
| QAT+PTQ | -91.61% ↓ | -0.0286% ↓ | +0.0534% ↑ | -75.14% ↓ | ✅ Mostly Better |
| QAT+QAT | -91.85% ↓ | +77.9809% ↑ | +58.4430% ↑ | -74.46% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.1778
- **Precision**: 0.1773
- **Recall**: 0.9971
- **F1-Score**: 0.3010
- **Normal Recall** (of actual normal, % predicted as normal): 0.0010
- **Normal Precision** (of predicted normal, % actually normal): 0.6093
- **Avg Latency**: 1.91 ms
- **Samples/sec**: 52260.26
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
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 202427.80
- **Compression Ratio**: 11.54x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9638
- **Precision**: 0.8677
- **Recall**: 0.9390
- **F1-Score**: 0.9020
- **Normal Recall** (of actual normal, % predicted as normal): 0.9691
- **Normal Precision** (of predicted normal, % actually normal): 0.9866
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 209025.42
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
- **Normal Precision** (of predicted normal, % actually normal): 0.0000
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 210177.59
- **Compression Ratio**: 11.91x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9576
- **Precision**: 0.8514
- **Recall**: 0.9223
- **F1-Score**: 0.8854
- **Normal Recall** (of actual normal, % predicted as normal): 0.9653
- **Normal Precision** (of predicted normal, % actually normal): 0.9829
- **Avg Latency**: 0.49 ms
- **Samples/sec**: 204610.18
- **Compression Ratio**: 12.28x

