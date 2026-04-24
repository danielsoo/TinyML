# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-04-15T21:37:41.972762

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
| Keras (global_model.h5) | 0.4429 | 0.8135 | -0.3706 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_no | 0.9294 | 0.8982 | 0.0312 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_yes | 0.9158 | 0.8982 | 0.0175 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_no | 0.9229 | 0.8987 | 0.0242 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_yes | 0.9239 | 0.8985 | 0.0254 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_no | 0.9229 | 0.8987 | 0.0242 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_yes | 0.9243 | 0.8985 | 0.0258 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_no | 0.9226 | 0.8987 | 0.0239 | 0.265100 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_yes | 0.9243 | 0.8985 | 0.0258 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_no | 0.9209 | 0.8985 | 0.0223 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_yes | 0.9377 | 0.8982 | 0.0396 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_no | 0.9201 | 0.8984 | 0.0217 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_yes | 0.9348 | 0.8983 | 0.0365 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_no | 0.9196 | 0.8985 | 0.0211 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_yes | 0.9347 | 0.8982 | 0.0365 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_no | 0.9198 | 0.8984 | 0.0214 | 0.265100 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_yes | 0.9347 | 0.8982 | 0.0365 | 0.265100 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4504 | 0.8186 | -0.3682 | 0.262363 |
| 0.05 | 0.4504 | 0.8108 | -0.3604 | 0.268182 |
| 0.1 | 0.4504 | 0.7852 | -0.3348 | 0.276190 |
| 0.15 | 0.4504 | 0.6154 | -0.1650 | 0.284561 |
| 0.2 | 0.4504 | 0.4746 | -0.0242 | 0.293148 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
