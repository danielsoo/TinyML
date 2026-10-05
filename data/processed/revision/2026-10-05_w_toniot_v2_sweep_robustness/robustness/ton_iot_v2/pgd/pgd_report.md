# PGD Attack Report

- **Dataset:** ton_iot
- **Data path:** data/raw/TON_IoT
- **Max samples:** inf
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/models/best_text_alpha05_test.h5`
- **Attack type:** pgd
- **Epsilon used:** 0.1
- **Generated:** 2026-10-05T12:29:45.821032

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
- `data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/models/best_text_alpha05_test.h5`
- `data/processed/revision/2026-10-05_u_toniot_v2/baseline/models/a_centralized.h5`
- `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/ton2_near_iid/fp32.tflite`
- `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/ton2_near_iid/ptq_only.tflite`
- `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/ton2_near_iid/prune_ft_client_ptq.tflite`
- `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/ton2_near_iid/prune_ft_client_qat.tflite`
- `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/ton2_near_iid/prune_ft_pooled_qat.tflite`

전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| best_text_alpha05_test | 0.9910 | 0.2293 | 0.7618 | 0.204153 | 368.848267 |
| a_centralized | 0.9952 | 0.2296 | 0.7656 | 0.204153 | 368.848267 |
| fp32 | 0.9910 | 0.2293 | 0.7618 | 0.204153 | 368.848267 |
| ptq_only | 0.9840 | 0.2294 | 0.7547 | 0.204153 | 368.848267 |
| prune_ft_client_ptq | 0.8572 | 0.2295 | 0.6277 | 0.204153 | 368.848267 |
| prune_ft_client_qat | 0.9798 | 0.2297 | 0.7501 | 0.204153 | 368.848267 |
| prune_ft_pooled_qat | 0.9871 | 0.9112 | 0.0759 | 0.204153 | 368.848267 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9900 | 0.2314 | 0.7586 | 0.178801 |
| 0.05 | 0.9900 | 0.2314 | 0.7586 | 0.184260 |
| 0.1 | 0.9900 | 0.2312 | 0.7588 | 0.198121 |
| 0.15 | 0.9900 | 0.2308 | 0.7592 | 0.211221 |
| 0.2 | 0.9900 | 0.2302 | 0.7598 | 0.226789 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
