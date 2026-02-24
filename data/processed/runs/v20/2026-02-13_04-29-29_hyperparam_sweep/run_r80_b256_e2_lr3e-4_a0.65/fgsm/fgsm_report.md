# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-18T19:30:46.345623

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.5183 | 0.2324 | 0.2859 | 0.267619 | 148.702133 |
| Original (TFLite) | 0.8862 | 0.6983 | 0.1879 | 0.267619 | 148.702133 |
| QAT+Prune only | 0.5909 | 0.8841 | -0.2931 | 0.267619 | 148.702133 |
| QAT+PTQ | 0.5866 | 0.8832 | -0.2967 | 0.267619 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.267619 | 148.702133 |
| Compressed (PTQ) | 0.5866 | 0.8832 | -0.2967 | 0.267619 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.5274 | 0.2348 | 0.2926 | 0.261947 |
| 0.05 | 0.5274 | 0.2348 | 0.2926 | 0.265651 |
| 0.1 | 0.5274 | 0.2348 | 0.2926 | 0.270579 |
| 0.15 | 0.5274 | 0.2348 | 0.2926 | 0.275647 |
| 0.2 | 0.5274 | 0.2348 | 0.2926 | 0.281098 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
