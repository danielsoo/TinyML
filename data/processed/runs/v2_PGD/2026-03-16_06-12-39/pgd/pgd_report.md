# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-03-16T06:21:30.645708

## 실험 설정 (이 실험에 사용된 요소)

| 항목 | 값 |
|------|-----|
| **PGD top-N** | 4 |
| **PGD metric** | f1_score |
| **AT enabled** | True |
| **AT attack** | pgd |
| **AT epsilon** | 0.05 |
| **평가 모델 수** | 5 |

**평가한 모델:**
- `models/global_model.h5`
- `models/tflite/saved_model_pruned_qat.tflite`
- `models/tflite/saved_model_traditional_qat.tflite`
- `models/tflite/saved_model_qat_ptq.tflite`
- `models/tflite/saved_model_no_qat_ptq.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9163 | 0.8954 | 0.0209 | 0.263588 | 148.702133 |
| Compressed (QAT) | 0.9342 | 0.8834 | 0.0508 | 0.263588 | 148.702133 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9343 | 0.1814 | 0.7529 | 0.263588 | 148.702133 |
| QAT+PTQ | 0.9416 | 0.9220 | 0.0197 | 0.263588 | 148.702133 |
| noQAT+PTQ | 0.2256 | 0.1802 | 0.0454 | 0.263588 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9192 | 0.8958 | 0.0234 | 0.262154 |
| 0.05 | 0.9192 | 0.8962 | 0.0230 | 0.266690 |
| 0.1 | 0.9192 | 0.8964 | 0.0228 | 0.273029 |
| 0.15 | 0.9192 | 0.8962 | 0.0230 | 0.279820 |
| 0.2 | 0.9192 | 0.8958 | 0.0234 | 0.287108 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
