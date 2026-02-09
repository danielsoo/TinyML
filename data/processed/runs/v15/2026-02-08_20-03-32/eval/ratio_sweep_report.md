# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-08 20:05:18 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 1500000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total ratios evaluated: 11

## Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.4520 | 0.0000 | 0.0000 | 0.0000 | 0.4520 | 1.0000 |
| 90 | 10 | 245,820 | 221,238 | 24,582 | 0.5002 | 0.1585 | 0.9278 | 0.2707 | 0.4527 | 0.9826 |
| 80 | 20 | 236,085 | 188,868 | 47,217 | 0.5480 | 0.2978 | 0.9280 | 0.4509 | 0.4530 | 0.9618 |
| 70 | 30 | 157,390 | 110,173 | 47,217 | 0.5957 | 0.4211 | 0.9280 | 0.5793 | 0.4533 | 0.9362 |
| 60 | 40 | 118,040 | 70,824 | 47,216 | 0.6433 | 0.5310 | 0.9280 | 0.6755 | 0.4536 | 0.9043 |
| 50 | 50 | 94,434 | 47,217 | 47,217 | 0.6904 | 0.6291 | 0.9280 | 0.7498 | 0.4529 | 0.8628 |
| 40 | 60 | 78,695 | 31,478 | 47,217 | 0.7371 | 0.7171 | 0.9280 | 0.8090 | 0.4509 | 0.8067 |
| 30 | 70 | 67,450 | 20,235 | 47,215 | 0.7859 | 0.7987 | 0.9280 | 0.8585 | 0.4544 | 0.7300 |
| 20 | 80 | 59,020 | 11,804 | 47,216 | 0.8333 | 0.8719 | 0.9280 | 0.8991 | 0.4547 | 0.6121 |
| 10 | 90 | 52,460 | 5,246 | 47,214 | 0.8809 | 0.9390 | 0.9280 | 0.9335 | 0.4577 | 0.4138 |
| 0 | 100 | 47,217 | 0 | 47,217 | 0.9280 | 1.0000 | 0.9280 | 0.9626 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.4520
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.4520
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 245,820 (Normal=221,238, Attack=24,582)
- **Accuracy**: 0.5002
- **Precision**: 0.1585
- **Recall**: 0.9278
- **F1-Score**: 0.2707
- **Normal Recall** (of actual normal, predicted normal): 0.4527
- **Normal Precision** (of predicted normal, actual normal): 0.9826

### Normal 80% : Attack 20%

- **Test samples**: 236,085 (Normal=188,868, Attack=47,217)
- **Accuracy**: 0.5480
- **Precision**: 0.2978
- **Recall**: 0.9280
- **F1-Score**: 0.4509
- **Normal Recall** (of actual normal, predicted normal): 0.4530
- **Normal Precision** (of predicted normal, actual normal): 0.9618

### Normal 70% : Attack 30%

- **Test samples**: 157,390 (Normal=110,173, Attack=47,217)
- **Accuracy**: 0.5957
- **Precision**: 0.4211
- **Recall**: 0.9280
- **F1-Score**: 0.5793
- **Normal Recall** (of actual normal, predicted normal): 0.4533
- **Normal Precision** (of predicted normal, actual normal): 0.9362

### Normal 60% : Attack 40%

- **Test samples**: 118,040 (Normal=70,824, Attack=47,216)
- **Accuracy**: 0.6433
- **Precision**: 0.5310
- **Recall**: 0.9280
- **F1-Score**: 0.6755
- **Normal Recall** (of actual normal, predicted normal): 0.4536
- **Normal Precision** (of predicted normal, actual normal): 0.9043

### Normal 50% : Attack 50%

- **Test samples**: 94,434 (Normal=47,217, Attack=47,217)
- **Accuracy**: 0.6904
- **Precision**: 0.6291
- **Recall**: 0.9280
- **F1-Score**: 0.7498
- **Normal Recall** (of actual normal, predicted normal): 0.4529
- **Normal Precision** (of predicted normal, actual normal): 0.8628

### Normal 40% : Attack 60%

- **Test samples**: 78,695 (Normal=31,478, Attack=47,217)
- **Accuracy**: 0.7371
- **Precision**: 0.7171
- **Recall**: 0.9280
- **F1-Score**: 0.8090
- **Normal Recall** (of actual normal, predicted normal): 0.4509
- **Normal Precision** (of predicted normal, actual normal): 0.8067

### Normal 30% : Attack 70%

- **Test samples**: 67,450 (Normal=20,235, Attack=47,215)
- **Accuracy**: 0.7859
- **Precision**: 0.7987
- **Recall**: 0.9280
- **F1-Score**: 0.8585
- **Normal Recall** (of actual normal, predicted normal): 0.4544
- **Normal Precision** (of predicted normal, actual normal): 0.7300

### Normal 20% : Attack 80%

- **Test samples**: 59,020 (Normal=11,804, Attack=47,216)
- **Accuracy**: 0.8333
- **Precision**: 0.8719
- **Recall**: 0.9280
- **F1-Score**: 0.8991
- **Normal Recall** (of actual normal, predicted normal): 0.4547
- **Normal Precision** (of predicted normal, actual normal): 0.6121

### Normal 10% : Attack 90%

- **Test samples**: 52,460 (Normal=5,246, Attack=47,214)
- **Accuracy**: 0.8809
- **Precision**: 0.9390
- **Recall**: 0.9280
- **F1-Score**: 0.9335
- **Normal Recall** (of actual normal, predicted normal): 0.4577
- **Normal Precision** (of predicted normal, actual normal): 0.4138

### Normal 0% : Attack 100%

- **Test samples**: 47,217 (Normal=0, Attack=47,217)
- **Accuracy**: 0.9280
- **Precision**: 1.0000
- **Recall**: 0.9280
- **F1-Score**: 0.9626
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.4520 | 0.0000 | 0.4520 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.5002 | 0.2707 | 0.4527 | 0.9826 | 0.9278 | 0.1585 |
| 80 | 20 | 0.15 | 0.5480 | 0.4509 | 0.4530 | 0.9618 | 0.9280 | 0.2978 |
| 70 | 30 | 0.15 | 0.5957 | 0.5793 | 0.4533 | 0.9362 | 0.9280 | 0.4211 |
| 60 | 40 | 0.15 | 0.6433 | 0.6755 | 0.4536 | 0.9043 | 0.9280 | 0.5310 |
| 50 | 50 | 0.15 | 0.6904 | 0.7498 | 0.4529 | 0.8628 | 0.9280 | 0.6291 |
| 40 | 60 | 0.15 | 0.7371 | 0.8090 | 0.4509 | 0.8067 | 0.9280 | 0.7171 |
| 30 | 70 | 0.15 | 0.7859 | 0.8585 | 0.4544 | 0.7300 | 0.9280 | 0.7987 |
| 20 | 80 | 0.15 | 0.8333 | 0.8991 | 0.4547 | 0.6121 | 0.9280 | 0.8719 |
| 10 | 90 | 0.15 | 0.8809 | 0.9335 | 0.4577 | 0.4138 | 0.9280 | 0.9390 |
| 0 | 100 | 0.15 | 0.9280 | 0.9626 | 0.0000 | 0.0000 | 0.9280 | 1.0000 |

