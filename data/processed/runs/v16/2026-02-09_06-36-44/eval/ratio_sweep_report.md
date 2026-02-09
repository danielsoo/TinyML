# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-09 06:39:15 |

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
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8267 | 0.0000 | 0.0000 | 0.0000 | 0.8267 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.8318 | 0.3615 | 0.8892 | 0.5140 | 0.8255 | 0.9853 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8377 | 0.5594 | 0.8885 | 0.6866 | 0.8251 | 0.9673 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.8455 | 0.6876 | 0.8885 | 0.7753 | 0.8270 | 0.9454 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.8502 | 0.7716 | 0.8885 | 0.8259 | 0.8246 | 0.9173 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.8582 | 0.8377 | 0.8885 | 0.8623 | 0.8278 | 0.8813 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.8632 | 0.8841 | 0.8885 | 0.8863 | 0.8252 | 0.8315 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.8707 | 0.9239 | 0.8885 | 0.9059 | 0.8293 | 0.7612 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.8759 | 0.9532 | 0.8885 | 0.9197 | 0.8256 | 0.6493 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.8823 | 0.9788 | 0.8885 | 0.9315 | 0.8265 | 0.4517 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.8885 | 1.0000 | 0.8885 | 0.9410 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8267
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8267
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.8318
- **Precision**: 0.3615
- **Recall**: 0.8892
- **F1-Score**: 0.5140
- **Normal Recall** (of actual normal, predicted normal): 0.8255
- **Normal Precision** (of predicted normal, actual normal): 0.9853

### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8377
- **Precision**: 0.5594
- **Recall**: 0.8885
- **F1-Score**: 0.6866
- **Normal Recall** (of actual normal, predicted normal): 0.8251
- **Normal Precision** (of predicted normal, actual normal): 0.9673

### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.8455
- **Precision**: 0.6876
- **Recall**: 0.8885
- **F1-Score**: 0.7753
- **Normal Recall** (of actual normal, predicted normal): 0.8270
- **Normal Precision** (of predicted normal, actual normal): 0.9454

### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.8502
- **Precision**: 0.7716
- **Recall**: 0.8885
- **F1-Score**: 0.8259
- **Normal Recall** (of actual normal, predicted normal): 0.8246
- **Normal Precision** (of predicted normal, actual normal): 0.9173

### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.8582
- **Precision**: 0.8377
- **Recall**: 0.8885
- **F1-Score**: 0.8623
- **Normal Recall** (of actual normal, predicted normal): 0.8278
- **Normal Precision** (of predicted normal, actual normal): 0.8813

### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.8632
- **Precision**: 0.8841
- **Recall**: 0.8885
- **F1-Score**: 0.8863
- **Normal Recall** (of actual normal, predicted normal): 0.8252
- **Normal Precision** (of predicted normal, actual normal): 0.8315

### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.8707
- **Precision**: 0.9239
- **Recall**: 0.8885
- **F1-Score**: 0.9059
- **Normal Recall** (of actual normal, predicted normal): 0.8293
- **Normal Precision** (of predicted normal, actual normal): 0.7612

### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.8759
- **Precision**: 0.9532
- **Recall**: 0.8885
- **F1-Score**: 0.9197
- **Normal Recall** (of actual normal, predicted normal): 0.8256
- **Normal Precision** (of predicted normal, actual normal): 0.6493

### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.8823
- **Precision**: 0.9788
- **Recall**: 0.8885
- **F1-Score**: 0.9315
- **Normal Recall** (of actual normal, predicted normal): 0.8265
- **Normal Precision** (of predicted normal, actual normal): 0.4517

### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.8885
- **Precision**: 1.0000
- **Recall**: 0.8885
- **F1-Score**: 0.9410
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.8267 | 0.0000 | 0.8267 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.8318 | 0.5140 | 0.8255 | 0.9853 | 0.8892 | 0.3615 |
| 80 | 20 | 0.15 | 0.8377 | 0.6866 | 0.8251 | 0.9673 | 0.8885 | 0.5594 |
| 70 | 30 | 0.15 | 0.8455 | 0.7753 | 0.8270 | 0.9454 | 0.8885 | 0.6876 |
| 60 | 40 | 0.15 | 0.8502 | 0.8259 | 0.8246 | 0.9173 | 0.8885 | 0.7716 |
| 50 | 50 | 0.15 | 0.8582 | 0.8623 | 0.8278 | 0.8813 | 0.8885 | 0.8377 |
| 40 | 60 | 0.15 | 0.8632 | 0.8863 | 0.8252 | 0.8315 | 0.8885 | 0.8841 |
| 30 | 70 | 0.15 | 0.8707 | 0.9059 | 0.8293 | 0.7612 | 0.8885 | 0.9239 |
| 20 | 80 | 0.15 | 0.8759 | 0.9197 | 0.8256 | 0.6493 | 0.8885 | 0.9532 |
| 10 | 90 | 0.15 | 0.8823 | 0.9315 | 0.8265 | 0.4517 | 0.8885 | 0.9788 |
| 0 | 100 | 0.15 | 0.8885 | 0.9410 | 0.0000 | 0.0000 | 0.8885 | 1.0000 |

