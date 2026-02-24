# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T14:39:56.063170

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.5668 | 0.3065 | 0.2604 | 0.265610 | 148.702133 |
| Original (TFLite) | 0.7138 | 0.8630 | -0.1492 | 0.265610 | 148.702133 |
| QAT+Prune only | 0.8065 | 0.4850 | 0.3215 | 0.265610 | 148.702133 |
| QAT+PTQ | 0.8085 | 0.4884 | 0.3201 | 0.265610 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.265610 | 148.702133 |
| Compressed (PTQ) | 0.8085 | 0.4884 | 0.3201 | 0.265610 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.5734 | 0.3122 | 0.2612 | 0.261758 |
| 0.05 | 0.5734 | 0.3120 | 0.2614 | 0.264619 |
| 0.1 | 0.5734 | 0.3118 | 0.2616 | 0.268574 |
| 0.15 | 0.5734 | 0.3120 | 0.2614 | 0.272931 |
| 0.2 | 0.5734 | 0.3120 | 0.2614 | 0.277705 |

## Epsilon tuning

- **Best epsilon:** 0.1000
- **Target success rate:** 0.50
