# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T06:00:05.801056

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9212 | 0.8627 | 0.0585 | 0.259363 | 148.702133 |
| Original (TFLite) | 0.8789 | 0.7878 | 0.0911 | 0.259363 | 148.702133 |
| QAT+Prune only | 0.7827 | 0.7998 | -0.0170 | 0.259363 | 148.702133 |
| QAT+PTQ | 0.7834 | 0.7997 | -0.0163 | 0.259363 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259363 | 148.702133 |
| Compressed (PTQ) | 0.7834 | 0.7997 | -0.0163 | 0.259363 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9232 | 0.8730 | 0.0502 | 0.261348 |
| 0.05 | 0.9232 | 0.8722 | 0.0510 | 0.261803 |
| 0.1 | 0.9232 | 0.8672 | 0.0560 | 0.262436 |
| 0.15 | 0.9232 | 0.8598 | 0.0634 | 0.263136 |
| 0.2 | 0.9232 | 0.8602 | 0.0630 | 0.263942 |

## Epsilon tuning

- **Best epsilon:** 0.1500
- **Target success rate:** 0.50
