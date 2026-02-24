# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-13T07:36:00.402366

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.5714 | 0.1785 | 0.3928 | 0.266482 | 148.702133 |
| Original (TFLite) | 0.9054 | 0.5270 | 0.3785 | 0.266482 | 148.702133 |
| QAT+Prune only | 0.8334 | 0.8746 | -0.0411 | 0.266482 | 148.702133 |
| QAT+PTQ | 0.8334 | 0.8741 | -0.0408 | 0.266482 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.266482 | 148.702133 |
| Compressed (PTQ) | 0.8334 | 0.8741 | -0.0408 | 0.266482 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.5826 | 0.1802 | 0.4024 | 0.261823 |
| 0.05 | 0.5826 | 0.1802 | 0.4024 | 0.264937 |
| 0.1 | 0.5826 | 0.1802 | 0.4024 | 0.269349 |
| 0.15 | 0.5826 | 0.1802 | 0.4024 | 0.274143 |
| 0.2 | 0.5826 | 0.1802 | 0.4024 | 0.279462 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
