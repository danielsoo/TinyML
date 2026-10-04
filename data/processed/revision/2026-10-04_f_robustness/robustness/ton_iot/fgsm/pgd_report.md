# PGD Attack Report

- **Dataset:** ton_iot
- **Data path:** data/raw/TON_IoT
- **Max samples:** inf
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `data/processed/revision/2026-10-04_d_toniot_float/baseline/models/b_fl.h5`
- **Attack type:** fgsm
- **Epsilon used:** 0.1
- **Generated:** 2026-10-04T12:04:31.310660

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
| b_fl | 0.9909 | 0.2298 | 0.7611 | 0.204321 | 208.338669 |
| a_centralized | 0.9920 | 0.2296 | 0.7624 | 0.204321 | 208.338669 |
| fp32 | 0.9909 | 0.2298 | 0.7611 | 0.204321 | 208.338669 |
| ptq_only | 0.9754 | 0.2297 | 0.7457 | 0.204321 | 208.338669 |
| prune_ft_client_ptq | 0.8609 | 0.2295 | 0.6314 | 0.204321 | 208.338669 |
| prune_ft_client_qat | 0.9808 | 0.2298 | 0.7510 | 0.204321 | 208.338669 |
| prune_ft_pooled_qat | 0.9873 | 0.5615 | 0.4258 | 0.204321 | 208.338669 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9898 | 0.2316 | 0.7582 | 0.172007 |
| 0.05 | 0.9898 | 0.2316 | 0.7582 | 0.178780 |
| 0.1 | 0.9898 | 0.2316 | 0.7582 | 0.199183 |
| 0.15 | 0.9898 | 0.2316 | 0.7582 | 0.221271 |
| 0.2 | 0.9898 | 0.2316 | 0.7582 | 0.244405 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
