# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v1_PGD |
| **Run (datetime)** | 2026-03-14_16-52-20 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-03-14 16:57:25 |

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
| Baseline | 0.7797 | 204,880 | 0.1760 | 0.2904 | 0.0089 | 0.8058 | 3.27 |
| Traditional+PTQ | 0.1850 | 170,015 | 0.2121 | 0.3019 | 0.0503 | 0.9998 | 1.39 |
| Traditional+QAT | 0.1672 | 170,015 | 0.9683 | 0.9092 | 0.9758 | 0.9859 | 1.29 |
| QAT+PTQ | 0.1850 | 170,015 | 0.9477 | 0.8624 | 0.9449 | 0.9917 | 1.29 |
| QAT+QAT | 0.1672 | 170,015 | 0.9532 | 0.8743 | 0.9528 | 0.9904 | 1.50 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -76.27% ↓ | +3.6133% ↑ | +1.1515% ↑ | -57.61% ↓ | ✅ All Improved |
| Traditional+QAT | -78.56% ↓ | +79.2322% ↑ | +61.8798% ↑ | -60.62% ↓ | ✅ All Improved |
| QAT+PTQ | -76.28% ↓ | +77.1778% ↑ | +57.2063% ↑ | -60.61% ↓ | ✅ All Improved |
| QAT+QAT | -78.56% ↓ | +77.7227% ↑ | +58.3882% ↑ | -54.10% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.1760
- **Precision**: 0.1702
- **Recall**: 0.9896
- **F1-Score**: 0.2904
- **Normal Recall** (of actual normal, % predicted as normal): 0.0089
- **Normal Precision** (of predicted normal, % actually normal): 0.8058
- **Avg Latency**: 3.27 ms
- **Samples/sec**: 30560.48
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,968 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.2121
- **Precision**: 0.1778
- **Recall**: 1.0000
- **F1-Score**: 0.3019
- **Normal Recall** (of actual normal, % predicted as normal): 0.0503
- **Normal Precision** (of predicted normal, % actually normal): 0.9998
- **Avg Latency**: 1.39 ms
- **Samples/sec**: 72090.61
- **Compression Ratio**: 4.21x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9683
- **Precision**: 0.8876
- **Recall**: 0.9318
- **F1-Score**: 0.9092
- **Normal Recall** (of actual normal, % predicted as normal): 0.9758
- **Normal Precision** (of predicted normal, % actually normal): 0.9859
- **Avg Latency**: 1.29 ms
- **Samples/sec**: 77596.14
- **Compression Ratio**: 4.66x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,944 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9477
- **Precision**: 0.7818
- **Recall**: 0.9616
- **F1-Score**: 0.8624
- **Normal Recall** (of actual normal, % predicted as normal): 0.9449
- **Normal Precision** (of predicted normal, % actually normal): 0.9917
- **Avg Latency**: 1.29 ms
- **Samples/sec**: 77588.96
- **Compression Ratio**: 4.22x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9532
- **Precision**: 0.8059
- **Recall**: 0.9552
- **F1-Score**: 0.8743
- **Normal Recall** (of actual normal, % predicted as normal): 0.9528
- **Normal Precision** (of predicted normal, % actually normal): 0.9904
- **Avg Latency**: 1.50 ms
- **Samples/sec**: 66574.14
- **Compression Ratio**: 4.66x

