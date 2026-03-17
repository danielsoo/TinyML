# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/MachineLearningCVE
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-03-17T09:28:00.326618

## 실험 설정 (이 실험에 사용된 요소)

| 항목 | 값 |
|------|-----|
| **PGD top-N** | 4 |
| **PGD metric** | f1_score |
| **AT enabled** | True |
| **AT attack** | pgd |
| **AT epsilon** | 0.05 |
| **평가 모델 수** | 49 |

**평가한 모델:**
- `/mnt/c/Users/mistf/TinyML/models/global_model.h5`
- `models/tflite/no_qat_distill_none_prune_none_ptq_no.tflite`
- `models/tflite/no_qat_distill_none_prune_none_ptq_yes.tflite`
- `models/tflite/no_qat_distill_none_prune_10x5_ptq_no.tflite`
- `models/tflite/no_qat_distill_none_prune_10x5_ptq_yes.tflite`
- `models/tflite/no_qat_distill_none_prune_10x2_ptq_no.tflite`
- `models/tflite/no_qat_distill_none_prune_10x2_ptq_yes.tflite`
- `models/tflite/no_qat_distill_none_prune_5x10_ptq_no.tflite`
- `models/tflite/no_qat_distill_none_prune_5x10_ptq_yes.tflite`
- `models/tflite/no_qat_distill_direct_prune_none_ptq_no.tflite`
- `models/tflite/no_qat_distill_direct_prune_none_ptq_yes.tflite`
- `models/tflite/no_qat_distill_direct_prune_10x5_ptq_no.tflite`
- `models/tflite/no_qat_distill_direct_prune_10x5_ptq_yes.tflite`
- `models/tflite/no_qat_distill_direct_prune_10x2_ptq_no.tflite`
- `models/tflite/no_qat_distill_direct_prune_10x2_ptq_yes.tflite`
- `models/tflite/no_qat_distill_direct_prune_5x10_ptq_no.tflite`
- `models/tflite/no_qat_distill_direct_prune_5x10_ptq_yes.tflite`
- `models/tflite/no_qat_distill_progressive_prune_none_ptq_no.tflite`
- `models/tflite/no_qat_distill_progressive_prune_none_ptq_yes.tflite`
- `models/tflite/no_qat_distill_progressive_prune_10x5_ptq_no.tflite`
- `models/tflite/no_qat_distill_progressive_prune_10x5_ptq_yes.tflite`
- `models/tflite/no_qat_distill_progressive_prune_10x2_ptq_no.tflite`
- `models/tflite/no_qat_distill_progressive_prune_10x2_ptq_yes.tflite`
- `models/tflite/no_qat_distill_progressive_prune_5x10_ptq_no.tflite`
- `models/tflite/no_qat_distill_progressive_prune_5x10_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_none_prune_none_ptq_no.tflite`
- `models/tflite/yes_qat_distill_none_prune_none_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_none_prune_10x5_ptq_no.tflite`
- `models/tflite/yes_qat_distill_none_prune_10x5_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_none_prune_10x2_ptq_no.tflite`
- `models/tflite/yes_qat_distill_none_prune_10x2_ptq_yes.tflite`
- `models/tflite/yes_qat_distill_none_prune_5x10_ptq_no.tflite`
- `models/tflite/yes_qat_distill_none_prune_5x10_ptq_yes.tflite`
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
| Keras (global_model.h5) | 0.7595 | 0.7307 | 0.0288 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_none_ptq_no | 0.4850 | 0.1814 | 0.3036 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_none_ptq_yes | 0.4783 | 0.1846 | 0.2938 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_10x5_ptq_no | 0.2842 | 0.1784 | 0.1057 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_10x5_ptq_yes | 0.3266 | 0.1785 | 0.1480 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_10x2_ptq_no | 0.2880 | 0.1784 | 0.1096 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_10x2_ptq_yes | 0.2955 | 0.1784 | 0.1171 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_5x10_ptq_no | 0.2825 | 0.1784 | 0.1041 | 0.265467 | 148.702133 |
| no_qat_distill_none_prune_5x10_ptq_yes | 0.3196 | 0.1785 | 0.1411 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_none_ptq_no | 0.8982 | 0.2569 | 0.6412 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_none_ptq_yes | 0.9165 | 0.1923 | 0.7243 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_10x5_ptq_no | 0.1920 | 0.1784 | 0.0135 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_10x5_ptq_yes | 0.1885 | 0.1784 | 0.0100 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_10x2_ptq_no | 0.1959 | 0.1784 | 0.0175 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_10x2_ptq_yes | 0.1948 | 0.1784 | 0.0163 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_5x10_ptq_no | 0.1925 | 0.1784 | 0.0140 | 0.265467 | 148.702133 |
| no_qat_distill_direct_prune_5x10_ptq_yes | 0.1902 | 0.1784 | 0.0118 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_none_ptq_no | 0.9218 | 0.1964 | 0.7255 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_none_ptq_yes | 0.9382 | 0.2110 | 0.7272 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_10x5_ptq_no | 0.1822 | 0.1784 | 0.0039 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_10x5_ptq_yes | 0.1809 | 0.1784 | 0.0026 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_10x2_ptq_no | 0.1812 | 0.1784 | 0.0029 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_10x2_ptq_yes | 0.1807 | 0.1784 | 0.0024 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_5x10_ptq_no | 0.1810 | 0.1784 | 0.0026 | 0.265467 | 148.702133 |
| no_qat_distill_progressive_prune_5x10_ptq_yes | 0.1815 | 0.1784 | 0.0032 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_none_ptq_no | 0.7595 | 0.7307 | 0.0288 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_none_ptq_yes | 0.7508 | 0.7470 | 0.0037 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_10x5_ptq_no | 0.8153 | 0.8626 | -0.0473 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_10x5_ptq_yes | 0.6769 | 0.8190 | -0.1421 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_10x2_ptq_no | 0.8410 | 0.8374 | 0.0037 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_10x2_ptq_yes | 0.8609 | 0.8921 | -0.0313 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_5x10_ptq_no | 0.8694 | 0.8771 | -0.0077 | 0.265467 | 148.702133 |
| yes_qat_distill_none_prune_5x10_ptq_yes | 0.7711 | 0.8673 | -0.0963 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_no | 0.9294 | 0.8988 | 0.0307 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_none_ptq_yes | 0.9158 | 0.8987 | 0.0171 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_no | 0.9228 | 0.8990 | 0.0238 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_10x5_ptq_yes | 0.9234 | 0.8990 | 0.0245 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_no | 0.9232 | 0.8990 | 0.0243 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_10x2_ptq_yes | 0.9235 | 0.8990 | 0.0245 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_no | 0.9243 | 0.8990 | 0.0254 | 0.265467 | 148.702133 |
| yes_qat_distill_direct_prune_5x10_ptq_yes | 0.9245 | 0.8990 | 0.0255 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_no | 0.9209 | 0.8992 | 0.0216 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_none_ptq_yes | 0.9377 | 0.8988 | 0.0390 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_no | 0.9207 | 0.8990 | 0.0217 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_10x5_ptq_yes | 0.9348 | 0.8988 | 0.0361 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_no | 0.9200 | 0.8985 | 0.0215 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_10x2_ptq_yes | 0.9347 | 0.8988 | 0.0360 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_no | 0.9201 | 0.8985 | 0.0216 | 0.265467 | 148.702133 |
| yes_qat_distill_progressive_prune_5x10_ptq_yes | 0.9348 | 0.8988 | 0.0361 | 0.265467 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.7568 | 0.7688 | -0.0120 | 0.262528 |
| 0.05 | 0.7568 | 0.7234 | 0.0334 | 0.268667 |
| 0.1 | 0.7568 | 0.6866 | 0.0702 | 0.276934 |
| 0.15 | 0.7568 | 0.6490 | 0.1078 | 0.285890 |
| 0.2 | 0.7568 | 0.6148 | 0.1420 | 0.295457 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
