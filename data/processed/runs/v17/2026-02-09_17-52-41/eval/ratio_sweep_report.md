# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-09 17:55:06 |

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
| 100 | 0 | 100,000 | 100,000 | 0 | 0.6104 | 0.0000 | 0.0000 | 0.0000 | 0.6104 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.6440 | 0.2115 | 0.9386 | 0.3453 | 0.6113 | 0.9890 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.6769 | 0.3766 | 0.9390 | 0.5376 | 0.6114 | 0.9756 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.7109 | 0.5098 | 0.9390 | 0.6608 | 0.6131 | 0.9591 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.7422 | 0.6168 | 0.9390 | 0.7445 | 0.6111 | 0.9376 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.7771 | 0.7094 | 0.9390 | 0.8082 | 0.6153 | 0.9097 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.8082 | 0.7841 | 0.9390 | 0.8546 | 0.6121 | 0.8699 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.8408 | 0.8494 | 0.9390 | 0.8920 | 0.6117 | 0.8111 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.8739 | 0.9068 | 0.9390 | 0.9226 | 0.6139 | 0.7154 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9063 | 0.9561 | 0.9390 | 0.9475 | 0.6121 | 0.5270 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9390 | 1.0000 | 0.9390 | 0.9685 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.6104
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.6104
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.6440
- **Precision**: 0.2115
- **Recall**: 0.9386
- **F1-Score**: 0.3453
- **Normal Recall** (of actual normal, predicted normal): 0.6113
- **Normal Precision** (of predicted normal, actual normal): 0.9890

### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.6769
- **Precision**: 0.3766
- **Recall**: 0.9390
- **F1-Score**: 0.5376
- **Normal Recall** (of actual normal, predicted normal): 0.6114
- **Normal Precision** (of predicted normal, actual normal): 0.9756

### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.7109
- **Precision**: 0.5098
- **Recall**: 0.9390
- **F1-Score**: 0.6608
- **Normal Recall** (of actual normal, predicted normal): 0.6131
- **Normal Precision** (of predicted normal, actual normal): 0.9591

### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.7422
- **Precision**: 0.6168
- **Recall**: 0.9390
- **F1-Score**: 0.7445
- **Normal Recall** (of actual normal, predicted normal): 0.6111
- **Normal Precision** (of predicted normal, actual normal): 0.9376

### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.7771
- **Precision**: 0.7094
- **Recall**: 0.9390
- **F1-Score**: 0.8082
- **Normal Recall** (of actual normal, predicted normal): 0.6153
- **Normal Precision** (of predicted normal, actual normal): 0.9097

### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.8082
- **Precision**: 0.7841
- **Recall**: 0.9390
- **F1-Score**: 0.8546
- **Normal Recall** (of actual normal, predicted normal): 0.6121
- **Normal Precision** (of predicted normal, actual normal): 0.8699

### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.8408
- **Precision**: 0.8494
- **Recall**: 0.9390
- **F1-Score**: 0.8920
- **Normal Recall** (of actual normal, predicted normal): 0.6117
- **Normal Precision** (of predicted normal, actual normal): 0.8111

### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.8739
- **Precision**: 0.9068
- **Recall**: 0.9390
- **F1-Score**: 0.9226
- **Normal Recall** (of actual normal, predicted normal): 0.6139
- **Normal Precision** (of predicted normal, actual normal): 0.7154

### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9063
- **Precision**: 0.9561
- **Recall**: 0.9390
- **F1-Score**: 0.9475
- **Normal Recall** (of actual normal, predicted normal): 0.6121
- **Normal Precision** (of predicted normal, actual normal): 0.5270

### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9390
- **Precision**: 1.0000
- **Recall**: 0.9390
- **F1-Score**: 0.9685
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.6104 | 0.0000 | 0.6104 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.6440 | 0.3453 | 0.6113 | 0.9890 | 0.9386 | 0.2115 |
| 80 | 20 | 0.15 | 0.6769 | 0.5376 | 0.6114 | 0.9756 | 0.9390 | 0.3766 |
| 70 | 30 | 0.15 | 0.7109 | 0.6608 | 0.6131 | 0.9591 | 0.9390 | 0.5098 |
| 60 | 40 | 0.15 | 0.7422 | 0.7445 | 0.6111 | 0.9376 | 0.9390 | 0.6168 |
| 50 | 50 | 0.15 | 0.7771 | 0.8082 | 0.6153 | 0.9097 | 0.9390 | 0.7094 |
| 40 | 60 | 0.15 | 0.8082 | 0.8546 | 0.6121 | 0.8699 | 0.9390 | 0.7841 |
| 30 | 70 | 0.15 | 0.8408 | 0.8920 | 0.6117 | 0.8111 | 0.9390 | 0.8494 |
| 20 | 80 | 0.15 | 0.8739 | 0.9226 | 0.6139 | 0.7154 | 0.9390 | 0.9068 |
| 10 | 90 | 0.15 | 0.9063 | 0.9475 | 0.6121 | 0.5270 | 0.9390 | 0.9561 |
| 0 | 100 | 0.15 | 0.9390 | 0.9685 | 0.0000 | 0.0000 | 0.9390 | 1.0000 |

