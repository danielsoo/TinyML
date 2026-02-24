# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-18T11:49:17.725514

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9338 | 0.8856 | 0.0483 | 0.259220 | 148.702133 |
| Original (TFLite) | 0.9274 | 0.8883 | 0.0391 | 0.259220 | 148.702133 |
| QAT+Prune only | 0.8270 | 0.8920 | -0.0649 | 0.259220 | 148.702133 |
| QAT+PTQ | 0.8256 | 0.8924 | -0.0668 | 0.259220 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259220 | 148.702133 |
| Compressed (PTQ) | 0.8256 | 0.8924 | -0.0668 | 0.259220 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9364 | 0.8954 | 0.0410 | 0.261338 |
| 0.05 | 0.9364 | 0.8936 | 0.0428 | 0.261730 |
| 0.1 | 0.9364 | 0.8930 | 0.0434 | 0.262276 |
| 0.15 | 0.9364 | 0.8914 | 0.0450 | 0.262849 |
| 0.2 | 0.9364 | 0.8896 | 0.0468 | 0.263511 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
