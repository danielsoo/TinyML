# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-04-16T20:34:07.896929

## 실험 설정 (이 실험에 사용된 요소)

| 항목 | 값 |
|------|-----|
| **PGD top-N** | 4 |
| **PGD metric** | f1_score |
| **AT enabled** | True |
| **AT attack** | pgd |
| **AT epsilon** | 0.05 |
| **평가 모델 수** | 17 |

**평가한 모델:**
- `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- `models/tflite/yes_qat_distill_direct_prune_none_ptq_no.tflite`
- `models/tflite/yes_qat_distill_direct_prune_none_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_direct_prune_10x5_ptq_no.tflite`
- `models/tflite/yes_qat_distill_direct_prune_10x5_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_direct_prune_10x2_ptq_no.tflite`
- `models/tflite/yes_qat_distill_direct_prune_10x2_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_direct_prune_5x10_ptq_no.tflite`
- `models/tflite/yes_qat_distill_direct_prune_5x10_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_none_ptq_no.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_none_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_10x5_ptq_no.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_10x5_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_10x2_ptq_no.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_10x2_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_5x10_ptq_no.tflite`
- `models/tflite/yes_qat_distill_progressive_prune_5x10_ptq_yes.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.7045 | 0.6371 | 0.0674 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_no | 0.9294 | 0.8984 | 0.0310 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_yes | 0.9158 | 0.8983 | 0.0175 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_no | 0.9231 | 0.8982 | 0.0249 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_yes | 0.9236 | 0.8983 | 0.0253 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_no | 0.9240 | 0.8982 | 0.0257 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_yes | 0.9245 | 0.8982 | 0.0262 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_no | 0.9232 | 0.8982 | 0.0249 | 0.265837 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_yes | 0.9236 | 0.8983 | 0.0253 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_no | 0.9209 | 0.8982 | 0.0226 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_yes | 0.9377 | 0.8984 | 0.0393 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_no | 0.9201 | 0.8985 | 0.0215 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_yes | 0.9307 | 0.8987 | 0.0321 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_no | 0.9199 | 0.8985 | 0.0214 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_yes | 0.9347 | 0.8988 | 0.0360 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_no | 0.9206 | 0.8985 | 0.0221 | 0.265837 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_yes | 0.9348 | 0.8987 | 0.0362 | 0.265837 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.6962 | 0.6668 | 0.0294 | 0.262525 |
| 0.05 | 0.6962 | 0.6300 | 0.0662 | 0.269008 |
| 0.1 | 0.6962 | 0.5728 | 0.1234 | 0.277564 |
| 0.15 | 0.6962 | 0.4976 | 0.1986 | 0.286269 |
| 0.2 | 0.6962 | 0.4608 | 0.2354 | 0.295445 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
