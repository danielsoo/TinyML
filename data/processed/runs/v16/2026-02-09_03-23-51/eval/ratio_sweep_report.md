# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-09 03:26:23 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total ratios evaluated: 11

## Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8990 | 0.0000 | 0.0000 | 0.0000 | 0.8990 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.8810 | 0.4430 | 0.7399 | 0.5542 | 0.8966 | 0.9688 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8647 | 0.6405 | 0.7378 | 0.6857 | 0.8964 | 0.9319 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.8492 | 0.7541 | 0.7378 | 0.7459 | 0.8969 | 0.8887 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.8327 | 0.8254 | 0.7378 | 0.7792 | 0.8960 | 0.8368 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.8174 | 0.8775 | 0.7378 | 0.8016 | 0.8970 | 0.7738 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.8015 | 0.9148 | 0.7378 | 0.8169 | 0.8970 | 0.6952 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.7855 | 0.9435 | 0.7378 | 0.8281 | 0.8969 | 0.5945 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.7699 | 0.9667 | 0.7378 | 0.8369 | 0.8983 | 0.4614 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.7544 | 0.9857 | 0.7378 | 0.8439 | 0.9036 | 0.2769 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.7378 | 1.0000 | 0.7378 | 0.8491 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8990
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8990
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.8810
- **Precision**: 0.4430
- **Recall**: 0.7399
- **F1-Score**: 0.5542
- **Normal Recall** (of actual normal, predicted normal): 0.8966
- **Normal Precision** (of predicted normal, actual normal): 0.9688

### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8647
- **Precision**: 0.6405
- **Recall**: 0.7378
- **F1-Score**: 0.6857
- **Normal Recall** (of actual normal, predicted normal): 0.8964
- **Normal Precision** (of predicted normal, actual normal): 0.9319

### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.8492
- **Precision**: 0.7541
- **Recall**: 0.7378
- **F1-Score**: 0.7459
- **Normal Recall** (of actual normal, predicted normal): 0.8969
- **Normal Precision** (of predicted normal, actual normal): 0.8887

### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.8327
- **Precision**: 0.8254
- **Recall**: 0.7378
- **F1-Score**: 0.7792
- **Normal Recall** (of actual normal, predicted normal): 0.8960
- **Normal Precision** (of predicted normal, actual normal): 0.8368

### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.8174
- **Precision**: 0.8775
- **Recall**: 0.7378
- **F1-Score**: 0.8016
- **Normal Recall** (of actual normal, predicted normal): 0.8970
- **Normal Precision** (of predicted normal, actual normal): 0.7738

### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.8015
- **Precision**: 0.9148
- **Recall**: 0.7378
- **F1-Score**: 0.8169
- **Normal Recall** (of actual normal, predicted normal): 0.8970
- **Normal Precision** (of predicted normal, actual normal): 0.6952

### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.7855
- **Precision**: 0.9435
- **Recall**: 0.7378
- **F1-Score**: 0.8281
- **Normal Recall** (of actual normal, predicted normal): 0.8969
- **Normal Precision** (of predicted normal, actual normal): 0.5945

### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.7699
- **Precision**: 0.9667
- **Recall**: 0.7378
- **F1-Score**: 0.8369
- **Normal Recall** (of actual normal, predicted normal): 0.8983
- **Normal Precision** (of predicted normal, actual normal): 0.4614

### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.7544
- **Precision**: 0.9857
- **Recall**: 0.7378
- **F1-Score**: 0.8439
- **Normal Recall** (of actual normal, predicted normal): 0.9036
- **Normal Precision** (of predicted normal, actual normal): 0.2769

### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.7378
- **Precision**: 1.0000
- **Recall**: 0.7378
- **F1-Score**: 0.8491
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.8990 | 0.0000 | 0.8990 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.8810 | 0.5542 | 0.8966 | 0.9688 | 0.7399 | 0.4430 |
| 80 | 20 | 0.15 | 0.8647 | 0.6857 | 0.8964 | 0.9319 | 0.7378 | 0.6405 |
| 70 | 30 | 0.15 | 0.8492 | 0.7459 | 0.8969 | 0.8887 | 0.7378 | 0.7541 |
| 60 | 40 | 0.15 | 0.8327 | 0.7792 | 0.8960 | 0.8368 | 0.7378 | 0.8254 |
| 50 | 50 | 0.15 | 0.8174 | 0.8016 | 0.8970 | 0.7738 | 0.7378 | 0.8775 |
| 40 | 60 | 0.15 | 0.8015 | 0.8169 | 0.8970 | 0.6952 | 0.7378 | 0.9148 |
| 30 | 70 | 0.15 | 0.7855 | 0.8281 | 0.8969 | 0.5945 | 0.7378 | 0.9435 |
| 20 | 80 | 0.15 | 0.7699 | 0.8369 | 0.8983 | 0.4614 | 0.7378 | 0.9667 |
| 10 | 90 | 0.15 | 0.7544 | 0.8439 | 0.9036 | 0.2769 | 0.7378 | 0.9857 |
| 0 | 100 | 0.15 | 0.7378 | 0.8491 | 0.0000 | 0.0000 | 0.7378 | 1.0000 |

