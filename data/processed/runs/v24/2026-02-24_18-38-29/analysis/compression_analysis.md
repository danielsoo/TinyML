# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v24 |
| **Run (datetime)** | 2026-02-24_18-38-29 |
| **Data Version** | cicids2017_max2000k_bal4.0 |
| **Generated** | 2026-02-24 18:39:27 |

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
| Baseline | 0.7797 | 204,880 | 0.1775 | 0.3015 | 0.0000 | 1.0000 | 2.02 |
| Traditional+PTQ | 0.0732 | 62,048 | 0.1775 | 0.3015 | 0.0000 | 0.0000 | 0.48 |
| Traditional+QAT | 0.0635 | 62,048 | 0.9630 | 0.8994 | 0.9696 | 0.9852 | 0.48 |
| QAT+PTQ | 0.0731 | 62,048 | 0.9475 | 0.8653 | 0.9470 | 0.9887 | 0.48 |
| QAT+QAT | 0.0635 | 62,048 | 0.9606 | 0.8950 | 0.9639 | 0.9879 | 0.48 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -90.62% ↓ | -0.0006% ↓ | -0.0002% ↓ | -76.43% ↓ | ✅ Mostly Better |
| Traditional+QAT | -91.85% ↓ | +78.5439% ↑ | +59.7895% ↑ | -75.99% ↓ | ✅ All Improved |
| QAT+PTQ | -90.62% ↓ | +76.9962% ↑ | +56.3779% ↑ | -76.20% ↓ | ✅ All Improved |
| QAT+QAT | -91.85% ↓ | +78.3081% ↑ | +59.3470% ↑ | -76.17% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.1775
- **Precision**: 0.1775
- **Recall**: 1.0000
- **F1-Score**: 0.3015
- **Normal Recall** (of actual normal, % predicted as normal): 0.0000
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 2.02 ms
- **Samples/sec**: 49540.58
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
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 210198.66
- **Compression Ratio**: 10.66x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9630
- **Precision**: 0.8686
- **Recall**: 0.9325
- **F1-Score**: 0.8994
- **Normal Recall** (of actual normal, % predicted as normal): 0.9696
- **Normal Precision** (of predicted normal, % actually normal): 0.9852
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 206311.07
- **Compression Ratio**: 12.28x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.0731 MB (76,688 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9475
- **Precision**: 0.7945
- **Recall**: 0.9500
- **F1-Score**: 0.8653
- **Normal Recall** (of actual normal, % predicted as normal): 0.9470
- **Normal Precision** (of predicted normal, % actually normal): 0.9887
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 208195.37
- **Compression Ratio**: 10.66x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.0635 MB (66,600 bytes)
- **Parameters**: 62,048
- **Accuracy**: 0.9606
- **Precision**: 0.8497
- **Recall**: 0.9454
- **F1-Score**: 0.8950
- **Normal Recall** (of actual normal, % predicted as normal): 0.9639
- **Normal Precision** (of predicted normal, % actually normal): 0.9879
- **Avg Latency**: 0.48 ms
- **Samples/sec**: 207885.80
- **Compression Ratio**: 12.28x

