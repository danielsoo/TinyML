# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated_scratch.yaml` |
| **Generated** | 2026-02-04 03:26:58 |

## Summary

Total ratios evaluated: 11

## Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9643 | 0.0000 | 0.0000 | 0.0000 | 0.9643 | 1.0000 |
| 90 | 10 | 245,890 | 221,301 | 24,589 | 0.9183 | 0.6110 | 0.5024 | 0.5514 | 0.9645 | 0.9458 |
| 80 | 20 | 235,990 | 188,792 | 47,198 | 0.8720 | 0.7793 | 0.5023 | 0.6109 | 0.9644 | 0.8857 |
| 70 | 30 | 157,323 | 110,126 | 47,197 | 0.8258 | 0.8583 | 0.5023 | 0.6338 | 0.9644 | 0.8189 |
| 60 | 40 | 117,995 | 70,797 | 47,198 | 0.7797 | 0.9044 | 0.5023 | 0.6459 | 0.9646 | 0.7441 |
| 50 | 50 | 94,396 | 47,198 | 47,198 | 0.7331 | 0.9329 | 0.5023 | 0.6530 | 0.9639 | 0.6595 |
| 40 | 60 | 78,661 | 31,464 | 47,197 | 0.6869 | 0.9540 | 0.5023 | 0.6581 | 0.9637 | 0.5635 |
| 30 | 70 | 67,422 | 20,226 | 47,196 | 0.6409 | 0.9702 | 0.5024 | 0.6620 | 0.9641 | 0.4536 |
| 20 | 80 | 58,995 | 11,799 | 47,196 | 0.5949 | 0.9831 | 0.5023 | 0.6649 | 0.9654 | 0.3266 |
| 10 | 90 | 52,440 | 5,244 | 47,196 | 0.5489 | 0.9930 | 0.5023 | 0.6672 | 0.9682 | 0.1777 |
| 0 | 100 | 47,198 | 0 | 47,198 | 0.5023 | 1.0000 | 0.5023 | 0.6687 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9643
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9643
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 245,890 (Normal=221,301, Attack=24,589)
- **Accuracy**: 0.9183
- **Precision**: 0.6110
- **Recall**: 0.5024
- **F1-Score**: 0.5514
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9645
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.9458

### Normal 80% : Attack 20%

- **Test samples**: 235,990 (Normal=188,792, Attack=47,198)
- **Accuracy**: 0.8720
- **Precision**: 0.7793
- **Recall**: 0.5023
- **F1-Score**: 0.6109
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9644
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.8857

### Normal 70% : Attack 30%

- **Test samples**: 157,323 (Normal=110,126, Attack=47,197)
- **Accuracy**: 0.8258
- **Precision**: 0.8583
- **Recall**: 0.5023
- **F1-Score**: 0.6338
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9644
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.8189

### Normal 60% : Attack 40%

- **Test samples**: 117,995 (Normal=70,797, Attack=47,198)
- **Accuracy**: 0.7797
- **Precision**: 0.9044
- **Recall**: 0.5023
- **F1-Score**: 0.6459
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9646
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.7441

### Normal 50% : Attack 50%

- **Test samples**: 94,396 (Normal=47,198, Attack=47,198)
- **Accuracy**: 0.7331
- **Precision**: 0.9329
- **Recall**: 0.5023
- **F1-Score**: 0.6530
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9639
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.6595

### Normal 40% : Attack 60%

- **Test samples**: 78,661 (Normal=31,464, Attack=47,197)
- **Accuracy**: 0.6869
- **Precision**: 0.9540
- **Recall**: 0.5023
- **F1-Score**: 0.6581
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9637
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.5635

### Normal 30% : Attack 70%

- **Test samples**: 67,422 (Normal=20,226, Attack=47,196)
- **Accuracy**: 0.6409
- **Precision**: 0.9702
- **Recall**: 0.5024
- **F1-Score**: 0.6620
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9641
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.4536

### Normal 20% : Attack 80%

- **Test samples**: 58,995 (Normal=11,799, Attack=47,196)
- **Accuracy**: 0.5949
- **Precision**: 0.9831
- **Recall**: 0.5023
- **F1-Score**: 0.6649
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9654
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.3266

### Normal 10% : Attack 90%

- **Test samples**: 52,440 (Normal=5,244, Attack=47,196)
- **Accuracy**: 0.5489
- **Precision**: 0.9930
- **Recall**: 0.5023
- **F1-Score**: 0.6672
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.9682
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.1777

### Normal 0% : Attack 100%

- **Test samples**: 47,198 (Normal=0, Attack=47,198)
- **Accuracy**: 0.5023
- **Precision**: 1.0000
- **Recall**: 0.5023
- **F1-Score**: 0.6687
- **Normal Recall** (정상인데 정상이라고 한 비율): 0.0000
- **Normal Precision** (정상이라고 한 것 중 정상인 비율): 0.0000

