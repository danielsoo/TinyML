# Compression Analysis Report

| Item | Value |
|------|----|
| **Version** | v2_PGD |
| **Run (datetime)** | 2026-03-16_06-12-39 |
| **Data Version** | cicids2017_bal4.0 |
| **Generated** | 2026-03-16 06:17:38 |

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
| Baseline | 0.7797 | 204,880 | 0.1705 | 0.2912 | 0.0002 | 0.9726 | 3.38 |
| Traditional+PTQ | 0.1850 | 170,015 | 0.2182 | 0.3035 | 0.0577 | 0.9998 | 1.16 |
| Traditional+QAT | 0.1672 | 170,015 | 0.9566 | 0.8846 | 0.9526 | 0.9949 | 1.53 |
| QAT+PTQ | 0.1850 | 170,015 | 0.9488 | 0.8656 | 0.9448 | 0.9931 | 1.43 |
| QAT+QAT | 0.1672 | 170,015 | 0.9659 | 0.9035 | 0.9716 | 0.9870 | 1.36 |

## 📊 Improvements vs Baseline

**Baseline:** Baseline

| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |
|-------|-------------|-----------------|-----------------|----------------|----------------|
| Baseline | - | - | - | - | **Baseline** |
| Traditional+PTQ | -76.27% ↓ | +4.7689% ↑ | +1.2365% ↑ | -65.69% ↓ | ✅ All Improved |
| Traditional+QAT | -78.56% ↓ | +78.6083% ↑ | +59.3416% ↑ | -54.74% ↓ | ✅ All Improved |
| QAT+PTQ | -76.28% ↓ | +77.8242% ↑ | +57.4378% ↑ | -57.75% ↓ | ✅ All Improved |
| QAT+QAT | -78.56% ↓ | +79.5337% ↑ | +61.2286% ↑ | -59.86% ↓ | ✅ All Improved |

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
- **Accuracy**: 0.1705
- **Precision**: 0.1704
- **Recall**: 1.0000
- **F1-Score**: 0.2912
- **Normal Recall** (of actual normal, % predicted as normal): 0.0002
- **Normal Precision** (of predicted normal, % actually normal): 0.9726
- **Avg Latency**: 3.38 ms
- **Samples/sec**: 29613.26
- **Compression Ratio**: 1.00x

### Traditional+PTQ

- **Model Path**: `models/tflite/saved_model_no_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,968 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.2182
- **Precision**: 0.1789
- **Recall**: 1.0000
- **F1-Score**: 0.3035
- **Normal Recall** (of actual normal, % predicted as normal): 0.0577
- **Normal Precision** (of predicted normal, % actually normal): 0.9998
- **Avg Latency**: 1.16 ms
- **Samples/sec**: 86299.00
- **Compression Ratio**: 4.21x

### Traditional+QAT

- **Model Path**: `models/tflite/saved_model_traditional_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9566
- **Precision**: 0.8086
- **Recall**: 0.9763
- **F1-Score**: 0.8846
- **Normal Recall** (of actual normal, % predicted as normal): 0.9526
- **Normal Precision** (of predicted normal, % actually normal): 0.9949
- **Avg Latency**: 1.53 ms
- **Samples/sec**: 65423.55
- **Compression Ratio**: 4.66x

### QAT+PTQ

- **Model Path**: `models/tflite/saved_model_qat_ptq.tflite`
- **File Size**: 0.1850 MB (193,944 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9488
- **Precision**: 0.7826
- **Recall**: 0.9682
- **F1-Score**: 0.8656
- **Normal Recall** (of actual normal, % predicted as normal): 0.9448
- **Normal Precision** (of predicted normal, % actually normal): 0.9931
- **Avg Latency**: 1.43 ms
- **Samples/sec**: 70093.15
- **Compression Ratio**: 4.22x

### QAT+QAT

- **Model Path**: `models/tflite/saved_model_pruned_qat.tflite`
- **File Size**: 0.1672 MB (175,288 bytes)
- **Parameters**: 170,015
- **Accuracy**: 0.9659
- **Precision**: 0.8715
- **Recall**: 0.9379
- **F1-Score**: 0.9035
- **Normal Recall** (of actual normal, % predicted as normal): 0.9716
- **Normal Precision** (of predicted normal, % actually normal): 0.9870
- **Avg Latency**: 1.36 ms
- **Samples/sec**: 73782.33
- **Compression Ratio**: 4.66x

