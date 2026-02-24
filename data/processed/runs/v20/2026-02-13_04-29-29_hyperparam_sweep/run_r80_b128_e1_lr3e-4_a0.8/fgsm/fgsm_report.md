# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-17T07:43:01.057421

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9148 | 0.8216 | 0.0932 | 0.259980 | 148.702133 |
| Original (TFLite) | 0.9157 | 0.8782 | 0.0376 | 0.259980 | 148.702133 |
| QAT+Prune only | 0.7947 | 0.6637 | 0.1310 | 0.259980 | 148.702133 |
| QAT+PTQ | 0.7951 | 0.6645 | 0.1305 | 0.259980 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259980 | 148.702133 |
| Compressed (PTQ) | 0.7951 | 0.6645 | 0.1305 | 0.259980 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9146 | 0.8198 | 0.0948 | 0.261393 |
| 0.05 | 0.9146 | 0.8198 | 0.0948 | 0.262097 |
| 0.1 | 0.9146 | 0.8198 | 0.0948 | 0.263076 |
| 0.15 | 0.9146 | 0.8198 | 0.0948 | 0.264166 |
| 0.2 | 0.9146 | 0.8198 | 0.0948 | 0.265349 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
