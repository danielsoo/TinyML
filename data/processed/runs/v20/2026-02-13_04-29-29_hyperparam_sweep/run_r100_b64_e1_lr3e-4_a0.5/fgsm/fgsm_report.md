# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-19T07:44:48.664319

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8511 | 0.2308 | 0.6203 | 0.261033 | 148.702133 |
| Original (TFLite) | 0.8871 | 0.8010 | 0.0862 | 0.261033 | 148.702133 |
| QAT+Prune only | 0.7803 | 0.7208 | 0.0595 | 0.261033 | 148.702133 |
| QAT+PTQ | 0.7803 | 0.7216 | 0.0588 | 0.261033 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.261033 | 148.702133 |
| Compressed (PTQ) | 0.7803 | 0.7216 | 0.0588 | 0.261033 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8542 | 0.2352 | 0.6190 | 0.261471 |
| 0.05 | 0.8542 | 0.2346 | 0.6196 | 0.262573 |
| 0.1 | 0.8542 | 0.2344 | 0.6198 | 0.264083 |
| 0.15 | 0.8542 | 0.2344 | 0.6198 | 0.265711 |
| 0.2 | 0.8542 | 0.2344 | 0.6198 | 0.267502 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
