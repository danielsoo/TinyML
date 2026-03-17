# 실험 설정 기록 (Experiment Record)

- **생성 시각:** 2026-03-16 17:48:31
- **전체 설정 파일:** 이 디렉터리의 `run_config.yaml`

## Data
| 항목 | 값 |
|------|-----|
| name | cicids2017 |

| path | data/raw/MachineLearningCVE |

| num_clients | 4 |

| max_samples | null |

| binary | True |

| use_smote | True |

| balance_ratio | 4.0 |


## Model
| 항목 | 값 |
|------|-----|
| name | mlp |


## Evaluation
| 항목 | 값 |
|------|-----|
| prediction_threshold | 0.3 |

| pgd_top_n | 4 |

| pgd_metric | f1_score |

| ratio_sweep_models | models/tflite/saved_model_original.tflite, models/tflite/saved_model_no_qat_ptq.tflite, models/tflite/saved_model_traditional_qat.tflite, models/tflite/saved_model_qat_ptq.tflite, models/tflite/saved_model_pruned_qat.tflite, models/tflite/saved_model_pruned_10x5_qat.tflite, models/tflite/saved_model_pruned_10x2_qat.tflite, models/tflite/saved_model_pruned_5x10_qat.tflite |


## Adversarial training (학습 직후·압축 직전 AT)
| 항목 | 값 |
|------|-----|
| enabled | True |

| attack | pgd |

| epochs | 3 |

| epsilon | 0.05 |

| adv_ratio | 0.5 |

| pgd_steps | 10 |

| pgd_alpha | null |


## Compression
| 항목 | 값 |
|------|-----|
| always_build_traditional | True |

| traditional_model_path | null |

| distillation_first | False |

| pruning_presets | [{'name': '10x5', 'ratio': 0.1}, {'name': '10x2', 'ratio': 0.02}, {'name': '5x10', 'ratio': 0.05}] |


## Federated
| 항목 | 값 |
|------|-----|
| num_rounds | 70 |

| fraction_fit | 1.0 |

| fraction_evaluate | 1.0 |

| local_epochs | 3 |

| batch_size | 128 |

| learning_rate | 0.001 |

| use_class_weights | True |

| use_focal_loss | True |

| focal_loss_alpha | 0.25 |

| server_momentum | 0.5 |

| server_learning_rate | 0.1 |

| lr_decay_type | cosine |

| lr_decay_rate | 0.95 |

| lr_drop_rate | 0.5 |

| lr_epochs_drop | 8 |

| lr_min | 0.0001 |

| use_qat | True |

| use_adversarial_training | False |

| min_fit_clients | 4 |

| min_evaluate_clients | 4 |

| min_available_clients | 4 |

