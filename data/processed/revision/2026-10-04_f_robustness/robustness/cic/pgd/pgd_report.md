# PGD Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** inf
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/b_fl.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-10-04T12:04:18.697622

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
| b_fl | 0.9426 | 0.5555 | 0.3871 | 0.290258 | 331.664429 |
| a_centralized | 0.9506 | 0.2900 | 0.6606 | 0.290258 | 331.664429 |
| fp32 | 0.9426 | 0.5555 | 0.3871 | 0.290258 | 331.664429 |
| ptq_only | 0.9386 | 0.6044 | 0.3342 | 0.290258 | 331.664429 |
| prune_ft_client_ptq | 0.9484 | 0.2392 | 0.7093 | 0.290258 | 331.664429 |
| prune_ft_client_qat | 0.9419 | 0.4556 | 0.4863 | 0.290258 | 331.664429 |
| prune_ft_pooled_qat | 0.9633 | 0.4618 | 0.5014 | 0.290258 | 331.664429 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9410 | 0.8006 | 0.1404 | 0.267395 |
| 0.05 | 0.9410 | 0.7216 | 0.2194 | 0.275398 |
| 0.1 | 0.9410 | 0.5548 | 0.3862 | 0.285967 |
| 0.15 | 0.9410 | 0.3768 | 0.5642 | 0.296916 |
| 0.2 | 0.9410 | 0.2650 | 0.6760 | 0.308376 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
