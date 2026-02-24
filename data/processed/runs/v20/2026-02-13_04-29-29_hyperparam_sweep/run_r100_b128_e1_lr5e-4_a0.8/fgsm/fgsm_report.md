# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-21T04:24:47.262858

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8911 | 0.8216 | 0.0696 | 0.260554 | 148.702133 |
| Original (TFLite) | 0.9222 | 0.8720 | 0.0502 | 0.260554 | 148.702133 |
| QAT+Prune only | 0.7899 | 0.7391 | 0.0508 | 0.260554 | 148.702133 |
| QAT+PTQ | 0.7893 | 0.7394 | 0.0498 | 0.260554 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.260554 | 148.702133 |
| Compressed (PTQ) | 0.7893 | 0.7394 | 0.0498 | 0.260554 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8916 | 0.8198 | 0.0718 | 0.261427 |
| 0.05 | 0.8916 | 0.8198 | 0.0718 | 0.262378 |
| 0.1 | 0.8916 | 0.8198 | 0.0718 | 0.263665 |
| 0.15 | 0.8916 | 0.8198 | 0.0718 | 0.265020 |
| 0.2 | 0.8916 | 0.8198 | 0.0718 | 0.266498 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
