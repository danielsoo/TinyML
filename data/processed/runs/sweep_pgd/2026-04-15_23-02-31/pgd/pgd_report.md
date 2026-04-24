# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-04-15T23:07:28.049908

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
| Keras (global_model.h5) | 0.7633 | 0.8043 | -0.0410 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_no | 0.9294 | 0.8978 | 0.0316 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_yes | 0.9158 | 0.8974 | 0.0184 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_no | 0.9226 | 0.8978 | 0.0249 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_yes | 0.9237 | 0.8980 | 0.0257 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_no | 0.9228 | 0.8978 | 0.0251 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_yes | 0.9237 | 0.8979 | 0.0259 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_no | 0.9226 | 0.8978 | 0.0249 | 0.265792 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_yes | 0.9234 | 0.8979 | 0.0256 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_no | 0.9209 | 0.8982 | 0.0226 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_yes | 0.9377 | 0.8979 | 0.0398 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_no | 0.9201 | 0.8980 | 0.0221 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_yes | 0.9307 | 0.8982 | 0.0326 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_no | 0.9200 | 0.8980 | 0.0220 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_yes | 0.9347 | 0.8982 | 0.0365 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_no | 0.9202 | 0.8980 | 0.0222 | 0.265792 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_yes | 0.9348 | 0.8982 | 0.0367 | 0.265792 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.7630 | 0.8432 | -0.0802 | 0.262599 |
| 0.05 | 0.7630 | 0.8064 | -0.0434 | 0.268929 |
| 0.1 | 0.7630 | 0.7520 | 0.0110 | 0.277336 |
| 0.15 | 0.7630 | 0.7036 | 0.0594 | 0.286079 |
| 0.2 | 0.7630 | 0.6252 | 0.1378 | 0.295269 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
