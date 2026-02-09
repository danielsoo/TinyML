# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-09 09:15:48 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | - |
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
| 100 | 0 | 100,000 | 100,000 | 0 | 0.6836 | 0.0000 | 0.0000 | 0.0000 | 0.6836 | 1.0000 |
| 90 | 10 | 460,810 | 414,729 | 46,081 | 0.7019 | 0.2327 | 0.8620 | 0.3664 | 0.6841 | 0.9781 |
| 80 | 20 | 425,865 | 340,692 | 85,173 | 0.7197 | 0.4055 | 0.8611 | 0.5513 | 0.6843 | 0.9517 |
| 70 | 30 | 283,910 | 198,737 | 85,173 | 0.7367 | 0.5383 | 0.8611 | 0.6624 | 0.6834 | 0.9199 |
| 60 | 40 | 212,930 | 127,758 | 85,172 | 0.7547 | 0.6449 | 0.8611 | 0.7375 | 0.6838 | 0.8807 |
| 50 | 50 | 170,346 | 85,173 | 85,173 | 0.7717 | 0.7305 | 0.8611 | 0.7904 | 0.6823 | 0.8308 |
| 40 | 60 | 141,955 | 56,782 | 85,173 | 0.7912 | 0.8046 | 0.8611 | 0.8319 | 0.6864 | 0.7671 |
| 30 | 70 | 121,672 | 36,501 | 85,171 | 0.8088 | 0.8651 | 0.8611 | 0.8631 | 0.6868 | 0.6794 |
| 20 | 80 | 106,465 | 21,293 | 85,172 | 0.8270 | 0.9176 | 0.8611 | 0.8884 | 0.6907 | 0.5542 |
| 10 | 90 | 94,630 | 9,463 | 85,167 | 0.8439 | 0.9614 | 0.8611 | 0.9085 | 0.6891 | 0.3554 |
| 0 | 100 | 85,173 | 0 | 85,173 | 0.8611 | 1.0000 | 0.8611 | 0.9254 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.6836
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.6836
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 460,810 (Normal=414,729, Attack=46,081)
- **Accuracy**: 0.7019
- **Precision**: 0.2327
- **Recall**: 0.8620
- **F1-Score**: 0.3664
- **Normal Recall** (of actual normal, predicted normal): 0.6841
- **Normal Precision** (of predicted normal, actual normal): 0.9781

### Normal 80% : Attack 20%

- **Test samples**: 425,865 (Normal=340,692, Attack=85,173)
- **Accuracy**: 0.7197
- **Precision**: 0.4055
- **Recall**: 0.8611
- **F1-Score**: 0.5513
- **Normal Recall** (of actual normal, predicted normal): 0.6843
- **Normal Precision** (of predicted normal, actual normal): 0.9517

### Normal 70% : Attack 30%

- **Test samples**: 283,910 (Normal=198,737, Attack=85,173)
- **Accuracy**: 0.7367
- **Precision**: 0.5383
- **Recall**: 0.8611
- **F1-Score**: 0.6624
- **Normal Recall** (of actual normal, predicted normal): 0.6834
- **Normal Precision** (of predicted normal, actual normal): 0.9199

### Normal 60% : Attack 40%

- **Test samples**: 212,930 (Normal=127,758, Attack=85,172)
- **Accuracy**: 0.7547
- **Precision**: 0.6449
- **Recall**: 0.8611
- **F1-Score**: 0.7375
- **Normal Recall** (of actual normal, predicted normal): 0.6838
- **Normal Precision** (of predicted normal, actual normal): 0.8807

### Normal 50% : Attack 50%

- **Test samples**: 170,346 (Normal=85,173, Attack=85,173)
- **Accuracy**: 0.7717
- **Precision**: 0.7305
- **Recall**: 0.8611
- **F1-Score**: 0.7904
- **Normal Recall** (of actual normal, predicted normal): 0.6823
- **Normal Precision** (of predicted normal, actual normal): 0.8308

### Normal 40% : Attack 60%

- **Test samples**: 141,955 (Normal=56,782, Attack=85,173)
- **Accuracy**: 0.7912
- **Precision**: 0.8046
- **Recall**: 0.8611
- **F1-Score**: 0.8319
- **Normal Recall** (of actual normal, predicted normal): 0.6864
- **Normal Precision** (of predicted normal, actual normal): 0.7671

### Normal 30% : Attack 70%

- **Test samples**: 121,672 (Normal=36,501, Attack=85,171)
- **Accuracy**: 0.8088
- **Precision**: 0.8651
- **Recall**: 0.8611
- **F1-Score**: 0.8631
- **Normal Recall** (of actual normal, predicted normal): 0.6868
- **Normal Precision** (of predicted normal, actual normal): 0.6794

### Normal 20% : Attack 80%

- **Test samples**: 106,465 (Normal=21,293, Attack=85,172)
- **Accuracy**: 0.8270
- **Precision**: 0.9176
- **Recall**: 0.8611
- **F1-Score**: 0.8884
- **Normal Recall** (of actual normal, predicted normal): 0.6907
- **Normal Precision** (of predicted normal, actual normal): 0.5542

### Normal 10% : Attack 90%

- **Test samples**: 94,630 (Normal=9,463, Attack=85,167)
- **Accuracy**: 0.8439
- **Precision**: 0.9614
- **Recall**: 0.8611
- **F1-Score**: 0.9085
- **Normal Recall** (of actual normal, predicted normal): 0.6891
- **Normal Precision** (of predicted normal, actual normal): 0.3554

### Normal 0% : Attack 100%

- **Test samples**: 85,173 (Normal=0, Attack=85,173)
- **Accuracy**: 0.8611
- **Precision**: 1.0000
- **Recall**: 0.8611
- **F1-Score**: 0.9254
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.6836 | 0.0000 | 0.6836 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.15 | 0.7019 | 0.3664 | 0.6841 | 0.9781 | 0.8620 | 0.2327 |
| 80 | 20 | 0.15 | 0.7197 | 0.5513 | 0.6843 | 0.9517 | 0.8611 | 0.4055 |
| 70 | 30 | 0.15 | 0.7367 | 0.6624 | 0.6834 | 0.9199 | 0.8611 | 0.5383 |
| 60 | 40 | 0.15 | 0.7547 | 0.7375 | 0.6838 | 0.8807 | 0.8611 | 0.6449 |
| 50 | 50 | 0.15 | 0.7717 | 0.7904 | 0.6823 | 0.8308 | 0.8611 | 0.7305 |
| 40 | 60 | 0.15 | 0.7912 | 0.8319 | 0.6864 | 0.7671 | 0.8611 | 0.8046 |
| 30 | 70 | 0.15 | 0.8088 | 0.8631 | 0.6868 | 0.6794 | 0.8611 | 0.8651 |
| 20 | 80 | 0.15 | 0.8270 | 0.8884 | 0.6907 | 0.5542 | 0.8611 | 0.9176 |
| 10 | 90 | 0.15 | 0.8439 | 0.9085 | 0.6891 | 0.3554 | 0.8611 | 0.9614 |
| 0 | 100 | 0.15 | 0.8611 | 0.9254 | 0.0000 | 0.0000 | 0.8611 | 1.0000 |

