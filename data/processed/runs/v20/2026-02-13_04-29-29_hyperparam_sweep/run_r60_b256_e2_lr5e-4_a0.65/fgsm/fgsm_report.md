# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T18:31:44.620990

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.5860 | 0.8188 | -0.2329 | 0.265713 | 148.702133 |
| Original (TFLite) | 0.5706 | 0.7240 | -0.1534 | 0.265713 | 148.702133 |
| QAT+Prune only | 0.8912 | 0.8605 | 0.0307 | 0.265713 | 148.702133 |
| QAT+PTQ | 0.8908 | 0.8611 | 0.0296 | 0.265713 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.265713 | 148.702133 |
| Compressed (PTQ) | 0.8908 | 0.8611 | 0.0296 | 0.265713 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.5984 | 0.8168 | -0.2184 | 0.261804 |
| 0.05 | 0.5984 | 0.8168 | -0.2184 | 0.264690 |
| 0.1 | 0.5984 | 0.8166 | -0.2182 | 0.268574 |
| 0.15 | 0.5984 | 0.8162 | -0.2178 | 0.272478 |
| 0.2 | 0.5984 | 0.8160 | -0.2176 | 0.276562 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
