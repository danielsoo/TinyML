# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-21T07:08:57.876959

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8216 | 0.8216 | 0.0000 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.6419 | 0.4931 | 0.1487 | 0.258160 | 148.702133 |
| QAT+Prune only | 0.6874 | 0.8404 | -0.1530 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.6867 | 0.8407 | -0.1540 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.258160 | 148.702133 |
| Compressed (PTQ) | 0.6867 | 0.8407 | -0.1540 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.05 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.1 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.15 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.2 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
