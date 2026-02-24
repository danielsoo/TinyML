# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T05:10:38.980451

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9312 | 0.8831 | 0.0481 | 0.259382 | 148.702133 |
| Original (TFLite) | 0.8826 | 0.8858 | -0.0032 | 0.259382 | 148.702133 |
| QAT+Prune only | 0.8396 | 0.7843 | 0.0554 | 0.259382 | 148.702133 |
| QAT+PTQ | 0.8393 | 0.7847 | 0.0546 | 0.259382 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259382 | 148.702133 |
| Compressed (PTQ) | 0.8393 | 0.7847 | 0.0546 | 0.259382 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9334 | 0.8812 | 0.0522 | 0.261342 |
| 0.05 | 0.9334 | 0.8810 | 0.0524 | 0.261795 |
| 0.1 | 0.9334 | 0.8806 | 0.0528 | 0.262445 |
| 0.15 | 0.9334 | 0.8802 | 0.0532 | 0.263158 |
| 0.2 | 0.9334 | 0.8792 | 0.0542 | 0.263976 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
