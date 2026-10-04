# PGD Attack Report

- **Dataset:** ton_iot
- **Data path:** data/raw/TON_IoT
- **Max samples:** inf
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `data/processed/revision/2026-10-04_d_toniot_float/baseline/models/b_fl.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-10-04T12:04:53.319479

## 실험 설정 (이 실험에 사용된 요소)

| 항목 | 값 |
|------|-----|
| **PGD top-N** | - |
| **PGD metric** | - |
| **AT enabled** | - |
| **AT attack** | - |
| **AT epsilon** | - |
| **평가 모델 수** | 7 |

**평가한 모델:**
- `data/processed/revision/2026-10-04_d_toniot_float/baseline/models/b_fl.h5`
- `data/processed/revision/2026-10-04_d_toniot_float/baseline/models/a_centralized.h5`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/ton_near_iid/fp32.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/ton_near_iid/ptq_only.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/ton_near_iid/prune_ft_client_ptq.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/ton_near_iid/prune_ft_client_qat.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/ton_near_iid/prune_ft_pooled_qat.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| b_fl | 0.9909 | 0.2295 | 0.7614 | 0.204068 | 208.338669 |
| a_centralized | 0.9920 | 0.2296 | 0.7623 | 0.204068 | 208.338669 |
| fp32 | 0.9909 | 0.2295 | 0.7614 | 0.204068 | 208.338669 |
| ptq_only | 0.9754 | 0.2295 | 0.7459 | 0.204068 | 208.338669 |
| prune_ft_client_ptq | 0.8609 | 0.2295 | 0.6314 | 0.204068 | 208.338669 |
| prune_ft_client_qat | 0.9808 | 0.2296 | 0.7512 | 0.204068 | 208.338669 |
| prune_ft_pooled_qat | 0.9873 | 0.5556 | 0.4316 | 0.204068 | 208.338669 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9898 | 0.2316 | 0.7582 | 0.172050 |
| 0.05 | 0.9898 | 0.2316 | 0.7582 | 0.179464 |
| 0.1 | 0.9898 | 0.2314 | 0.7584 | 0.199034 |
| 0.15 | 0.9898 | 0.2314 | 0.7584 | 0.215064 |
| 0.2 | 0.9898 | 0.2314 | 0.7584 | 0.228730 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
