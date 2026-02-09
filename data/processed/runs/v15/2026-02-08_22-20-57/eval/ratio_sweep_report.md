# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-08 22:22:58 |

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
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8318 | 0.0000 | 0.0000 | 0.0000 | 0.8318 | 1.0000 |
| 90 | 10 | 245,820 | 221,238 | 24,582 | 0.8081 | 0.2797 | 0.5832 | 0.3780 | 0.8331 | 0.9473 |
| 80 | 20 | 236,085 | 188,868 | 47,217 | 0.7836 | 0.4673 | 0.5854 | 0.5198 | 0.8332 | 0.8894 |
| 70 | 30 | 157,390 | 110,173 | 47,217 | 0.7589 | 0.6007 | 0.5854 | 0.5930 | 0.8333 | 0.8242 |
| 60 | 40 | 118,040 | 70,824 | 47,216 | 0.7344 | 0.7013 | 0.5854 | 0.6382 | 0.8338 | 0.7510 |
| 50 | 50 | 94,434 | 47,217 | 47,217 | 0.7080 | 0.7756 | 0.5854 | 0.6672 | 0.8306 | 0.6671 |
| 40 | 60 | 78,695 | 31,478 | 47,217 | 0.6836 | 0.8384 | 0.5854 | 0.6895 | 0.8308 | 0.5719 |
| 30 | 70 | 67,450 | 20,235 | 47,215 | 0.6588 | 0.8894 | 0.5854 | 0.7061 | 0.8301 | 0.4618 |
| 20 | 80 | 59,020 | 11,804 | 47,216 | 0.6351 | 0.9337 | 0.5854 | 0.7196 | 0.8338 | 0.3346 |
| 10 | 90 | 52,460 | 5,246 | 47,214 | 0.6110 | 0.9706 | 0.5854 | 0.7304 | 0.8406 | 0.1839 |
| 0 | 100 | 47,217 | 0 | 47,217 | 0.5854 | 1.0000 | 0.5854 | 0.7385 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8318
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8318
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 245,820 (Normal=221,238, Attack=24,582)
- **Accuracy**: 0.8081
- **Precision**: 0.2797
- **Recall**: 0.5832
- **F1-Score**: 0.3780
- **Normal Recall** (of actual normal, predicted normal): 0.8331
- **Normal Precision** (of predicted normal, actual normal): 0.9473

### Normal 80% : Attack 20%

- **Test samples**: 236,085 (Normal=188,868, Attack=47,217)
- **Accuracy**: 0.7836
- **Precision**: 0.4673
- **Recall**: 0.5854
- **F1-Score**: 0.5198
- **Normal Recall** (of actual normal, predicted normal): 0.8332
- **Normal Precision** (of predicted normal, actual normal): 0.8894

### Normal 70% : Attack 30%

- **Test samples**: 157,390 (Normal=110,173, Attack=47,217)
- **Accuracy**: 0.7589
- **Precision**: 0.6007
- **Recall**: 0.5854
- **F1-Score**: 0.5930
- **Normal Recall** (of actual normal, predicted normal): 0.8333
- **Normal Precision** (of predicted normal, actual normal): 0.8242

### Normal 60% : Attack 40%

- **Test samples**: 118,040 (Normal=70,824, Attack=47,216)
- **Accuracy**: 0.7344
- **Precision**: 0.7013
- **Recall**: 0.5854
- **F1-Score**: 0.6382
- **Normal Recall** (of actual normal, predicted normal): 0.8338
- **Normal Precision** (of predicted normal, actual normal): 0.7510

### Normal 50% : Attack 50%

- **Test samples**: 94,434 (Normal=47,217, Attack=47,217)
- **Accuracy**: 0.7080
- **Precision**: 0.7756
- **Recall**: 0.5854
- **F1-Score**: 0.6672
- **Normal Recall** (of actual normal, predicted normal): 0.8306
- **Normal Precision** (of predicted normal, actual normal): 0.6671

### Normal 40% : Attack 60%

- **Test samples**: 78,695 (Normal=31,478, Attack=47,217)
- **Accuracy**: 0.6836
- **Precision**: 0.8384
- **Recall**: 0.5854
- **F1-Score**: 0.6895
- **Normal Recall** (of actual normal, predicted normal): 0.8308
- **Normal Precision** (of predicted normal, actual normal): 0.5719

### Normal 30% : Attack 70%

- **Test samples**: 67,450 (Normal=20,235, Attack=47,215)
- **Accuracy**: 0.6588
- **Precision**: 0.8894
- **Recall**: 0.5854
- **F1-Score**: 0.7061
- **Normal Recall** (of actual normal, predicted normal): 0.8301
- **Normal Precision** (of predicted normal, actual normal): 0.4618

### Normal 20% : Attack 80%

- **Test samples**: 59,020 (Normal=11,804, Attack=47,216)
- **Accuracy**: 0.6351
- **Precision**: 0.9337
- **Recall**: 0.5854
- **F1-Score**: 0.7196
- **Normal Recall** (of actual normal, predicted normal): 0.8338
- **Normal Precision** (of predicted normal, actual normal): 0.3346

### Normal 10% : Attack 90%

- **Test samples**: 52,460 (Normal=5,246, Attack=47,214)
- **Accuracy**: 0.6110
- **Precision**: 0.9706
- **Recall**: 0.5854
- **F1-Score**: 0.7304
- **Normal Recall** (of actual normal, predicted normal): 0.8406
- **Normal Precision** (of predicted normal, actual normal): 0.1839

### Normal 0% : Attack 100%

- **Test samples**: 47,217 (Normal=0, Attack=47,217)
- **Accuracy**: 0.5854
- **Precision**: 1.0000
- **Recall**: 0.5854
- **F1-Score**: 0.7385
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.8318 | 0.0000 | 0.8318 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.8081 | 0.3780 | 0.8331 | 0.9473 | 0.5832 | 0.2797 |
| 80 | 20 | 0.15 | 0.7836 | 0.5198 | 0.8332 | 0.8894 | 0.5854 | 0.4673 |
| 70 | 30 | 0.15 | 0.7589 | 0.5930 | 0.8333 | 0.8242 | 0.5854 | 0.6007 |
| 60 | 40 | 0.15 | 0.7344 | 0.6382 | 0.8338 | 0.7510 | 0.5854 | 0.7013 |
| 50 | 50 | 0.15 | 0.7080 | 0.6672 | 0.8306 | 0.6671 | 0.5854 | 0.7756 |
| 40 | 60 | 0.15 | 0.6836 | 0.6895 | 0.8308 | 0.5719 | 0.5854 | 0.8384 |
| 30 | 70 | 0.15 | 0.6588 | 0.7061 | 0.8301 | 0.4618 | 0.5854 | 0.8894 |
| 20 | 80 | 0.15 | 0.6351 | 0.7196 | 0.8338 | 0.3346 | 0.5854 | 0.9337 |
| 10 | 90 | 0.15 | 0.6110 | 0.7304 | 0.8406 | 0.1839 | 0.5854 | 0.9706 |
| 0 | 100 | 0.15 | 0.5854 | 0.7385 | 0.0000 | 0.0000 | 0.5854 | 1.0000 |

