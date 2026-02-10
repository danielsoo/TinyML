# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-10 00:20:35 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **Aggregation strategy** | FedAvgM (momentum=0.5) |
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
| 100 | 0 | 100,000 | 100,000 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.9000 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8000 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7000 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4000 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3000 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

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

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.9000

### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.8000

### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.7000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.7000

### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.6000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.6000

### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.5000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.5000

### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.4000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.4000

### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.3000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.3000

### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.2000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.2000

### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.1000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.1000

### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
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

