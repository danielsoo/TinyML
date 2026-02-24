# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T17:15:43.020751

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8468 | 0.8216 | 0.0251 | 0.261767 | 148.702133 |
| Original (TFLite) | 0.9089 | 0.8683 | 0.0405 | 0.261767 | 148.702133 |
| QAT+Prune only | 0.7039 | 0.6090 | 0.0949 | 0.261767 | 148.702133 |
| QAT+PTQ | 0.7043 | 0.6108 | 0.0935 | 0.261767 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.261767 | 148.702133 |
| Compressed (PTQ) | 0.7043 | 0.6108 | 0.0935 | 0.261767 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8460 | 0.8198 | 0.0262 | 0.261512 |
| 0.05 | 0.8460 | 0.8198 | 0.0262 | 0.262948 |
| 0.1 | 0.8460 | 0.8198 | 0.0262 | 0.264909 |
| 0.15 | 0.8460 | 0.8198 | 0.0262 | 0.267147 |
| 0.2 | 0.8460 | 0.8198 | 0.0262 | 0.269670 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
