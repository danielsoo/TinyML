# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v12 |
| **Run (datetime)** | 2026-03-12_19-07-57 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-03-12 19:12:45 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Data path** | data/raw/MachineLearningCVE |
| **Max samples** | None |
| **Balance ratio** (normal:attack) | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Binary** | True |
| **Use SMOTE** | True |
| **Model** | mlp |
| **FL rounds** | 70 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Fraction fit** | 1.0 |
| **Fraction evaluate** | 1.0 |
| **Use class weights** | True |
| **Use focal loss** | True |
| **Focal loss alpha** | 0.25 |
| **Use distillation** | - |
| **Use QAT** | True |
| **Server momentum** | 0.5 |
| **Server learning rate** | 0.1 |
| **LR decay type** | cosine |
| **LR decay rate** | 0.95 |
| **LR drop rate** | 0.5 |
| **LR epochs drop** | 8 |
| **LR min** | 0.0001 |
| **Min fit clients** | 4 |
| **Min evaluate clients** | 4 |
| **Min available clients** | 4 |
| **Prediction threshold** | 0.5 |
| **Ratio sweep models** | - |
| **Always build traditional** | - |
| **Traditional model path** | null |

## Summary

Total stages analyzed: 5

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Baseline | 0.7797 | 204,880 | 0.7858 | 0.3065 | 0.8901 | 0.8572 | 3.43 |
| Traditional+PTQ | 0.0732 | 62,048 | 0.9172 | 0.7051 | 0.9862 | 0.9198 | 0.75 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9678 | 0.9090 | 0.9727 | 0.9883 | 0.90 |
| QAT+PTQ | 0.0731 | 62,048 | 0.9620 | 0.8872 | 0.9792 | 0.9751 | 0.78 |
| QAT+QAT | 0.0635 | 62,048 | 0.9668 | 0.9036 | 0.9782 | 0.9818 | 1.11 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -90.62% ↓ | +13.1386% ↑ | +39.8644% ↑ | -78.19% ↓ | ✅ All Improved |
| Traditional+QAT | -91.85% ↓ | +18.2031% ↑ | +60.2553% ↑ | -73.82% ↓ | ✅ All Improved |
| QAT+PTQ | -90.62% ↓ | +17.6176% ↑ | +58.0734% ↑ | -77.43% ↓ | ✅ All Improved |
| QAT+QAT | -91.85% ↓ | +18.1061% ↑ | +59.7089% ↑ | -67.69% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.7858
- **Precision**: 0.3417
- **Recall**: 0.2778
- **F1-Score**: 0.3065
- **Normal Recall** (of actual normal, % predicted as normal): 0.8901
- **Normal Precision** (of predicted normal, % actually normal): 0.8572
- **Avg Latency**: 3.43 ms
- **Samples/sec**: 29123.67
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.0732 MB (76,720 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9172
- **Precision**: 0.8960
- **Recall**: 0.5813
- **F1-Score**: 0.7051
- **Normal Recall** (of actual normal, % predicted as normal): 0.9862
- **Normal Precision** (of predicted normal, % actually normal): 0.9198
- **Avg Latency**: 0.75 ms
- **Samples/sec**: 133529.78
- **Compression Ratio**: 10.66x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9678
- **Precision**: 0.8767
- **Recall**: 0.9438
- **F1-Score**: 0.9090
- **Normal Recall** (of actual normal, % predicted as normal): 0.9727
- **Normal Precision** (of predicted normal, % actually normal): 0.9883
- **Avg Latency**: 0.90 ms
- **Samples/sec**: 111225.25
- **Compression Ratio**: 12.28x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0731 MB (76,688 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9620
- **Precision**: 0.8965
- **Recall**: 0.8781
- **F1-Score**: 0.8872
- **Normal Recall** (of actual normal, % predicted as normal): 0.9792
- **Normal Precision** (of predicted normal, % actually normal): 0.9751
- **Avg Latency**: 0.78 ms
- **Samples/sec**: 129023.75
- **Compression Ratio**: 10.66x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9668
- **Precision**: 0.8956
- **Recall**: 0.9117
- **F1-Score**: 0.9036
- **Normal Recall** (of actual normal, % predicted as normal): 0.9782
- **Normal Precision** (of predicted normal, % actually normal): 0.9818
- **Avg Latency**: 1.11 ms
- **Samples/sec**: 90124.50
- **Compression Ratio**: 12.28x

