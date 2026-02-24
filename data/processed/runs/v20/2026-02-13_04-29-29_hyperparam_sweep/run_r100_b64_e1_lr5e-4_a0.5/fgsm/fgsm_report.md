# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-19T12:17:11.114714

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8429 | 0.8216 | 0.0213 | 0.261791 | 148.702133 |
| Original (TFLite) | 0.9008 | 0.8216 | 0.0791 | 0.261791 | 148.702133 |
| QAT+Prune only | 0.7622 | 0.7129 | 0.0493 | 0.261791 | 148.702133 |
| QAT+PTQ | 0.7631 | 0.7137 | 0.0493 | 0.261791 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.261791 | 148.702133 |
| Compressed (PTQ) | 0.7631 | 0.7137 | 0.0493 | 0.261791 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8406 | 0.8198 | 0.0208 | 0.261528 |
| 0.05 | 0.8406 | 0.8198 | 0.0208 | 0.262968 |
| 0.1 | 0.8406 | 0.8198 | 0.0208 | 0.264955 |
| 0.15 | 0.8406 | 0.8198 | 0.0208 | 0.267219 |
| 0.2 | 0.8406 | 0.8198 | 0.0208 | 0.269761 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
