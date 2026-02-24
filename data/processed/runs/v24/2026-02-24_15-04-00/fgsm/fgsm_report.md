# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-24T15:05:58.932508

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8747 | 0.8216 | 0.0530 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.1787 | 0.1784 | 0.0003 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.1784 | 0.1784 | 0.0000 | 0.258160 | 148.702133 |
| saved_model_traditional_qat | 0.9535 | 0.2364 | 0.7171 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.9519 | 0.9158 | 0.0361 | 0.258160 | 148.702133 |
| Compressed (QAT) | 0.9657 | 0.9280 | 0.0377 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8790 | 0.8166 | 0.0624 | 0.261266 |
| 0.05 | 0.8790 | 0.8166 | 0.0624 | 0.261266 |
| 0.1 | 0.8790 | 0.8166 | 0.0624 | 0.261266 |
| 0.15 | 0.8790 | 0.8166 | 0.0624 | 0.261266 |
| 0.2 | 0.8790 | 0.8166 | 0.0624 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
