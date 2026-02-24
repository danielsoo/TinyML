# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-17T11:12:18.157417

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9309 | 0.8216 | 0.1093 | 0.259460 | 148.702133 |
| Original (TFLite) | 0.9063 | 0.8446 | 0.0617 | 0.259460 | 148.702133 |
| QAT+Prune only | 0.5333 | 0.2424 | 0.2908 | 0.259460 | 148.702133 |
| QAT+PTQ | 0.5340 | 0.2432 | 0.2908 | 0.259460 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259460 | 148.702133 |
| Compressed (PTQ) | 0.5340 | 0.2432 | 0.2908 | 0.259460 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9334 | 0.8198 | 0.1136 | 0.261342 |
| 0.05 | 0.9334 | 0.8198 | 0.1136 | 0.261828 |
| 0.1 | 0.9334 | 0.8198 | 0.1136 | 0.262509 |
| 0.15 | 0.9334 | 0.8198 | 0.1136 | 0.263274 |
| 0.2 | 0.9334 | 0.8198 | 0.1136 | 0.264145 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
