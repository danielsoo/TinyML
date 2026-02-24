# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-13T05:13:53.730593

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9357 | 0.8269 | 0.1088 | 0.259210 | 148.702133 |
| Original (TFLite) | 0.8998 | 0.8781 | 0.0216 | 0.259210 | 148.702133 |
| QAT+Prune only | 0.8297 | 0.7246 | 0.1051 | 0.259210 | 148.702133 |
| QAT+PTQ | 0.8292 | 0.7086 | 0.1206 | 0.259210 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259210 | 148.702133 |
| Compressed (PTQ) | 0.8292 | 0.7086 | 0.1206 | 0.259210 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9372 | 0.8242 | 0.1130 | 0.261336 |
| 0.05 | 0.9372 | 0.8242 | 0.1130 | 0.261735 |
| 0.1 | 0.9372 | 0.8242 | 0.1130 | 0.262296 |
| 0.15 | 0.9372 | 0.8242 | 0.1130 | 0.262868 |
| 0.2 | 0.9372 | 0.8242 | 0.1130 | 0.263482 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
