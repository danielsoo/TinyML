# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-08 19:43:40 |

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
| 100 | 0 | 100,000 | 100,000 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 245,820 | 221,238 | 24,582 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.9000 |
| 80 | 20 | 236,085 | 188,868 | 47,217 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8000 |
| 70 | 30 | 157,390 | 110,173 | 47,217 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7000 |
| 60 | 40 | 118,040 | 70,824 | 47,216 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 |
| 50 | 50 | 94,434 | 47,217 | 47,217 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 40 | 60 | 78,695 | 31,478 | 47,217 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4000 |
| 30 | 70 | 67,450 | 20,235 | 47,215 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3000 |
| 20 | 80 | 59,020 | 11,804 | 47,216 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 10 | 90 | 52,460 | 5,246 | 47,214 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |
| 0 | 100 | 47,217 | 0 | 47,217 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 1.0000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 245,820 (Normal=221,238, Attack=24,582)
- **Accuracy**: 0.9000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.9000

### Normal 80% : Attack 20%

- **Test samples**: 236,085 (Normal=188,868, Attack=47,217)
- **Accuracy**: 0.8000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.8000

### Normal 70% : Attack 30%

- **Test samples**: 157,390 (Normal=110,173, Attack=47,217)
- **Accuracy**: 0.7000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.7000

### Normal 60% : Attack 40%

- **Test samples**: 118,040 (Normal=70,824, Attack=47,216)
- **Accuracy**: 0.6000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.6000

### Normal 50% : Attack 50%

- **Test samples**: 94,434 (Normal=47,217, Attack=47,217)
- **Accuracy**: 0.5000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.5000

### Normal 40% : Attack 60%

- **Test samples**: 78,695 (Normal=31,478, Attack=47,217)
- **Accuracy**: 0.4000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.4000

### Normal 30% : Attack 70%

- **Test samples**: 67,450 (Normal=20,235, Attack=47,215)
- **Accuracy**: 0.3000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.3000

### Normal 20% : Attack 80%

- **Test samples**: 59,020 (Normal=11,804, Attack=47,216)
- **Accuracy**: 0.2000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.2000

### Normal 10% : Attack 90%

- **Test samples**: 52,460 (Normal=5,246, Attack=47,214)
- **Accuracy**: 0.1000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.1000

### Normal 0% : Attack 100%

- **Test samples**: 47,217 (Normal=0, Attack=47,217)
- **Accuracy**: 0.0000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.9000 | 0.0000 | 1.0000 | 0.9000 | 0.0000 | 0.0000 |
| 80 | 20 | 0.15 | 0.8000 | 0.0000 | 1.0000 | 0.8000 | 0.0000 | 0.0000 |
| 70 | 30 | 0.15 | 0.7000 | 0.0000 | 1.0000 | 0.7000 | 0.0000 | 0.0000 |
| 60 | 40 | 0.15 | 0.6000 | 0.0000 | 1.0000 | 0.6000 | 0.0000 | 0.0000 |
| 50 | 50 | 0.15 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.0000 | 0.0000 |
| 40 | 60 | 0.15 | 0.4000 | 0.0000 | 1.0000 | 0.4000 | 0.0000 | 0.0000 |
| 30 | 70 | 0.15 | 0.3000 | 0.0000 | 1.0000 | 0.3000 | 0.0000 | 0.0000 |
| 20 | 80 | 0.15 | 0.2000 | 0.0000 | 1.0000 | 0.2000 | 0.0000 | 0.0000 |
| 10 | 90 | 0.15 | 0.1000 | 0.0000 | 1.0000 | 0.1000 | 0.0000 | 0.0000 |
| 0 | 100 | 0.15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

