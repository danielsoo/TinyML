# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v1_PGD |
| **Run (datetime)** | 2026-03-14_11-37-42 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-03-14 11:42:47 |

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
| **Prediction threshold** | 0.3 |
| **Ratio sweep models** | 8 models |
| **Always build traditional** | True |
| **Traditional model path** | null |
| **Distillation first** | False |
| **Adversarial training (AT) enabled** | True |
| **AT attack** | pgd |
| **AT epsilon** | 0.05 |
| **AT epochs** | 3 |
| **AT adv_ratio** | 0.5 |
| **AT pgd_steps** | 10 |
| **AT pgd_alpha** | None |
| **PGD top-N models** | 4 |
| **PGD metric** | f1_score |

**전체 실험 설정:** 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Summary

Total stages analyzed: 5

| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |
|-------|-----------|------------|----------|----------|---------------|------------------|--------------|
| Baseline | 0.7797 | 204,880 | 0.1707 | 0.2912 | 0.0003 | 1.0000 | 3.08 |
| Traditional+PTQ | 0.1850 | 170,015 | 0.2157 | 0.3029 | 0.0547 | 0.9998 | 1.26 |
| Traditional+QAT | 0.1672 | 170,015 | 0.9682 | 0.9106 | 0.9718 | 0.9897 | 1.12 |
| QAT+PTQ | 0.1850 | 170,015 | 0.9490 | 0.8649 | 0.9469 | 0.9912 | 1.20 |
| QAT+QAT | 0.1672 | 170,015 | 0.9639 | 0.8968 | 0.9730 | 0.9833 | 1.18 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -76.27% ↓ | +4.5054% ↑ | +1.1652% ↑ | -59.20% ↓ | ✅ All Improved |
| Traditional+QAT | -78.56% ↓ | +79.7523% ↑ | +61.9351% ↑ | -63.48% ↓ | ✅ All Improved |
| QAT+PTQ | -76.28% ↓ | +77.8302% ↑ | +57.3715% ↑ | -60.95% ↓ | ✅ All Improved |
| QAT+QAT | -78.56% ↓ | +79.3266% ↑ | +60.5545% ↑ | -61.82% ↓ | ✅ All Improved |

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
| Traditional+PTQ | 4.21x | 76.3% |
| Traditional+QAT | 4.66x | 78.6% |
| QAT+PTQ | 4.22x | 76.3% |
| QAT+QAT | 4.66x | 78.6% |

## Detailed Metrics

### Baseline

- **Model Path**: `models/tflite/saved_model_original.tflite`
- **File Size**: 0.7797 MB (817,528 bytes)
- **Parameters**: 204,880
- **Accuracy**: 0.1707
- **Precision**: 0.1704
- **Recall**: 1.0000
- **F1-Score**: 0.2912
- **Normal Recall** (of actual normal, % predicted as normal): 0.0003
- **Normal Precision** (of predicted normal, % actually normal): 1.0000
- **Avg Latency**: 3.08 ms
- **Samples/sec**: 32484.02
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,968 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.2157
- **Precision**: 0.1785
- **Recall**: 1.0000
- **F1-Score**: 0.3029
- **Normal Recall** (of actual normal, % predicted as normal): 0.0547
- **Normal Precision** (of predicted normal, % actually normal): 0.9998
- **Avg Latency**: 1.26 ms
- **Samples/sec**: 79621.55
- **Compression Ratio**: 4.21x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9682
- **Precision**: 0.8736
- **Recall**: 0.9508
- **F1-Score**: 0.9106
- **Normal Recall** (of actual normal, % predicted as normal): 0.9718
- **Normal Precision** (of predicted normal, % actually normal): 0.9897
- **Avg Latency**: 1.12 ms
- **Samples/sec**: 88958.49
- **Compression Ratio**: 4.66x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,944 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9490
- **Precision**: 0.7875
- **Recall**: 0.9592
- **F1-Score**: 0.8649
- **Normal Recall** (of actual normal, % predicted as normal): 0.9469
- **Normal Precision** (of predicted normal, % actually normal): 0.9912
- **Avg Latency**: 1.20 ms
- **Samples/sec**: 83182.36
- **Compression Ratio**: 4.22x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9639
- **Precision**: 0.8750
- **Recall**: 0.9197
- **F1-Score**: 0.8968
- **Normal Recall** (of actual normal, % predicted as normal): 0.9730
- **Normal Precision** (of predicted normal, % actually normal): 0.9833
- **Avg Latency**: 1.18 ms
- **Samples/sec**: 85071.98
- **Compression Ratio**: 4.66x

