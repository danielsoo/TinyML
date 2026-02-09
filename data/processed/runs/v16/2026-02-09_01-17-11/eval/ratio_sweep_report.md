# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-09 01:21:00 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | None |
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
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8498 | 0.0000 | 0.0000 | 0.0000 | 0.8498 | 1.0000 |
| 90 | 10 | 460,810 | 414,729 | 46,081 | 0.8112 | 0.2539 | 0.4577 | 0.3266 | 0.8505 | 0.9338 |
| 80 | 20 | 425,865 | 340,692 | 85,173 | 0.7720 | 0.4336 | 0.4576 | 0.4453 | 0.8506 | 0.8625 |
| 70 | 30 | 283,910 | 198,737 | 85,173 | 0.7325 | 0.5672 | 0.4576 | 0.5065 | 0.8504 | 0.7853 |
| 60 | 40 | 212,930 | 127,758 | 85,172 | 0.6931 | 0.6705 | 0.4576 | 0.5439 | 0.8501 | 0.7016 |
| 50 | 50 | 170,346 | 85,173 | 85,173 | 0.6527 | 0.7505 | 0.4576 | 0.5685 | 0.8479 | 0.6099 |
| 40 | 60 | 141,955 | 56,782 | 85,173 | 0.6148 | 0.8213 | 0.4576 | 0.5877 | 0.8507 | 0.5111 |
| 30 | 70 | 121,672 | 36,501 | 85,171 | 0.5741 | 0.8740 | 0.4576 | 0.6007 | 0.8461 | 0.4006 |
| 20 | 80 | 106,465 | 21,293 | 85,172 | 0.5363 | 0.9250 | 0.4576 | 0.6122 | 0.8515 | 0.2818 |
| 10 | 90 | 94,630 | 9,463 | 85,167 | 0.4964 | 0.9640 | 0.4575 | 0.6206 | 0.8463 | 0.1477 |
| 0 | 100 | 85,173 | 0 | 85,173 | 0.4576 | 1.0000 | 0.4576 | 0.6278 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8498
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8498
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 460,810 (Normal=414,729, Attack=46,081)
- **Accuracy**: 0.8112
- **Precision**: 0.2539
- **Recall**: 0.4577
- **F1-Score**: 0.3266
- **Normal Recall** (of actual normal, predicted normal): 0.8505
- **Normal Precision** (of predicted normal, actual normal): 0.9338

### Normal 80% : Attack 20%

- **Test samples**: 425,865 (Normal=340,692, Attack=85,173)
- **Accuracy**: 0.7720
- **Precision**: 0.4336
- **Recall**: 0.4576
- **F1-Score**: 0.4453
- **Normal Recall** (of actual normal, predicted normal): 0.8506
- **Normal Precision** (of predicted normal, actual normal): 0.8625

### Normal 70% : Attack 30%

- **Test samples**: 283,910 (Normal=198,737, Attack=85,173)
- **Accuracy**: 0.7325
- **Precision**: 0.5672
- **Recall**: 0.4576
- **F1-Score**: 0.5065
- **Normal Recall** (of actual normal, predicted normal): 0.8504
- **Normal Precision** (of predicted normal, actual normal): 0.7853

### Normal 60% : Attack 40%

- **Test samples**: 212,930 (Normal=127,758, Attack=85,172)
- **Accuracy**: 0.6931
- **Precision**: 0.6705
- **Recall**: 0.4576
- **F1-Score**: 0.5439
- **Normal Recall** (of actual normal, predicted normal): 0.8501
- **Normal Precision** (of predicted normal, actual normal): 0.7016

### Normal 50% : Attack 50%

- **Test samples**: 170,346 (Normal=85,173, Attack=85,173)
- **Accuracy**: 0.6527
- **Precision**: 0.7505
- **Recall**: 0.4576
- **F1-Score**: 0.5685
- **Normal Recall** (of actual normal, predicted normal): 0.8479
- **Normal Precision** (of predicted normal, actual normal): 0.6099

### Normal 40% : Attack 60%

- **Test samples**: 141,955 (Normal=56,782, Attack=85,173)
- **Accuracy**: 0.6148
- **Precision**: 0.8213
- **Recall**: 0.4576
- **F1-Score**: 0.5877
- **Normal Recall** (of actual normal, predicted normal): 0.8507
- **Normal Precision** (of predicted normal, actual normal): 0.5111

### Normal 30% : Attack 70%

- **Test samples**: 121,672 (Normal=36,501, Attack=85,171)
- **Accuracy**: 0.5741
- **Precision**: 0.8740
- **Recall**: 0.4576
- **F1-Score**: 0.6007
- **Normal Recall** (of actual normal, predicted normal): 0.8461
- **Normal Precision** (of predicted normal, actual normal): 0.4006

### Normal 20% : Attack 80%

- **Test samples**: 106,465 (Normal=21,293, Attack=85,172)
- **Accuracy**: 0.5363
- **Precision**: 0.9250
- **Recall**: 0.4576
- **F1-Score**: 0.6122
- **Normal Recall** (of actual normal, predicted normal): 0.8515
- **Normal Precision** (of predicted normal, actual normal): 0.2818

### Normal 10% : Attack 90%

- **Test samples**: 94,630 (Normal=9,463, Attack=85,167)
- **Accuracy**: 0.4964
- **Precision**: 0.9640
- **Recall**: 0.4575
- **F1-Score**: 0.6206
- **Normal Recall** (of actual normal, predicted normal): 0.8463
- **Normal Precision** (of predicted normal, actual normal): 0.1477

### Normal 0% : Attack 100%

- **Test samples**: 85,173 (Normal=0, Attack=85,173)
- **Accuracy**: 0.4576
- **Precision**: 1.0000
- **Recall**: 0.4576
- **F1-Score**: 0.6278
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.8498 | 0.0000 | 0.8498 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.8112 | 0.3266 | 0.8505 | 0.9338 | 0.4577 | 0.2539 |
| 80 | 20 | 0.15 | 0.7720 | 0.4453 | 0.8506 | 0.8625 | 0.4576 | 0.4336 |
| 70 | 30 | 0.15 | 0.7325 | 0.5065 | 0.8504 | 0.7853 | 0.4576 | 0.5672 |
| 60 | 40 | 0.15 | 0.6931 | 0.5439 | 0.8501 | 0.7016 | 0.4576 | 0.6705 |
| 50 | 50 | 0.15 | 0.6527 | 0.5685 | 0.8479 | 0.6099 | 0.4576 | 0.7505 |
| 40 | 60 | 0.15 | 0.6148 | 0.5877 | 0.8507 | 0.5111 | 0.4576 | 0.8213 |
| 30 | 70 | 0.15 | 0.5741 | 0.6007 | 0.8461 | 0.4006 | 0.4576 | 0.8740 |
| 20 | 80 | 0.15 | 0.5363 | 0.6122 | 0.8515 | 0.2818 | 0.4576 | 0.9250 |
| 10 | 90 | 0.15 | 0.4964 | 0.6206 | 0.8463 | 0.1477 | 0.4575 | 0.9640 |
| 0 | 100 | 0.15 | 0.4576 | 0.6278 | 0.0000 | 0.0000 | 0.4576 | 1.0000 |

