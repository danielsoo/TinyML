# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-14T09:44:28.903624

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9351 | 0.8898 | 0.0452 | 0.259149 | 148.702133 |
| Original (TFLite) | 0.9007 | 0.8662 | 0.0345 | 0.259149 | 148.702133 |
| QAT+Prune only | 0.8108 | 0.7976 | 0.0132 | 0.259149 | 148.702133 |
| QAT+PTQ | 0.8118 | 0.7996 | 0.0122 | 0.259149 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259149 | 148.702133 |
| Compressed (PTQ) | 0.8118 | 0.7996 | 0.0122 | 0.259149 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9370 | 0.8918 | 0.0452 | 0.261338 |
| 0.05 | 0.9370 | 0.8918 | 0.0452 | 0.261728 |
| 0.1 | 0.9370 | 0.8914 | 0.0456 | 0.262232 |
| 0.15 | 0.9370 | 0.8910 | 0.0460 | 0.262767 |
| 0.2 | 0.9370 | 0.8908 | 0.0462 | 0.263382 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
