# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-22T19:44:49.595195

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4607 | 0.2097 | 0.2510 | 0.269172 | 148.702133 |
| Original (TFLite) | 0.8875 | 0.5326 | 0.3549 | 0.269172 | 148.702133 |
| QAT+Prune only | 0.3255 | 0.2364 | 0.0891 | 0.269172 | 148.702133 |
| QAT+PTQ | 0.3251 | 0.2417 | 0.0833 | 0.269172 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.269172 | 148.702133 |
| Compressed (PTQ) | 0.3251 | 0.2417 | 0.0833 | 0.269172 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4626 | 0.2134 | 0.2492 | 0.262124 |
| 0.05 | 0.4626 | 0.2132 | 0.2494 | 0.266501 |
| 0.1 | 0.4626 | 0.2132 | 0.2494 | 0.272224 |
| 0.15 | 0.4626 | 0.2134 | 0.2492 | 0.278212 |
| 0.2 | 0.4626 | 0.2134 | 0.2492 | 0.284991 |

## Epsilon tuning

- **Best epsilon:** 0.0500
- **Target success rate:** 0.50
