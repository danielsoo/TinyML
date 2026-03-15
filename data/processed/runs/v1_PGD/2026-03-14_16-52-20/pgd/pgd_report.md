# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-03-14T17:01:19.228477

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
- `models/tflite/saved_model_traditional_qat.tflite`
- `models/tflite/saved_model_pruned_qat.tflite`
- `models/tflite/saved_model_qat_ptq.tflite`
- `models/tflite/saved_model_no_qat_ptq.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9274 | 0.9001 | 0.0272 | 0.263745 | 148.702133 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9236 | 0.1816 | 0.7420 | 0.263745 | 148.702133 |
| Compressed (QAT) | 0.9226 | 0.8814 | 0.0412 | 0.263745 | 148.702133 |
| QAT+PTQ | 0.9446 | 0.9059 | 0.0388 | 0.263745 | 148.702133 |
| noQAT+PTQ | 0.2206 | 0.1800 | 0.0406 | 0.263745 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9350 | 0.9054 | 0.0296 | 0.262176 |
| 0.05 | 0.9350 | 0.9064 | 0.0286 | 0.266827 |
| 0.1 | 0.9350 | 0.9070 | 0.0280 | 0.273346 |
| 0.15 | 0.9350 | 0.9102 | 0.0248 | 0.280111 |
| 0.2 | 0.9350 | 0.9098 | 0.0252 | 0.287554 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
