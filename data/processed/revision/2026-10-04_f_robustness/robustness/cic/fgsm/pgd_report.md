# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** inf
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/b_fl.h5`
- **Attack type:** fgsm
- **Epsilon used:** 0.1
- **Generated:** 2026-10-04T12:02:02.315250

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
- `data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/b_fl.h5`
- `data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/a_centralized.h5`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/cic_near_iid/fp32.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/cic_near_iid/ptq_only.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/cic_near_iid/prune_ft_client_ptq.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/cic_near_iid/prune_ft_client_qat.tflite`
- `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/cic_near_iid/prune_ft_pooled_qat.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| b_fl | 0.9426 | 0.6858 | 0.2568 | 0.291511 | 331.664429 |
| a_centralized | 0.9506 | 0.3075 | 0.6431 | 0.291511 | 331.664429 |
| fp32 | 0.9426 | 0.6858 | 0.2568 | 0.291511 | 331.664429 |
| ptq_only | 0.9386 | 0.7038 | 0.2349 | 0.291511 | 331.664429 |
| prune_ft_client_ptq | 0.9484 | 0.3248 | 0.6236 | 0.291511 | 331.664429 |
| prune_ft_client_qat | 0.9419 | 0.4429 | 0.4990 | 0.291511 | 331.664429 |
| prune_ft_pooled_qat | 0.9633 | 0.4474 | 0.5159 | 0.291511 | 331.664429 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9410 | 0.8084 | 0.1326 | 0.267445 |
| 0.05 | 0.9410 | 0.7734 | 0.1676 | 0.275823 |
| 0.1 | 0.9410 | 0.6862 | 0.2548 | 0.287257 |
| 0.15 | 0.9410 | 0.6314 | 0.3096 | 0.299910 |
| 0.2 | 0.9410 | 0.5546 | 0.3864 | 0.313716 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
