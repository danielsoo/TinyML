# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-20T23:10:41.878390

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9157 | 0.8222 | 0.0935 | 0.259800 | 148.702133 |
| Original (TFLite) | 0.9331 | 0.9081 | 0.0250 | 0.259800 | 148.702133 |
| QAT+Prune only | 0.7106 | 0.7149 | -0.0043 | 0.259800 | 148.702133 |
| QAT+PTQ | 0.7107 | 0.7140 | -0.0033 | 0.259800 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259800 | 148.702133 |
| Compressed (PTQ) | 0.7107 | 0.7140 | -0.0033 | 0.259800 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9176 | 0.8204 | 0.0972 | 0.261373 |
| 0.05 | 0.9176 | 0.8204 | 0.0972 | 0.262007 |
| 0.1 | 0.9176 | 0.8204 | 0.0972 | 0.262887 |
| 0.15 | 0.9176 | 0.8204 | 0.0972 | 0.263882 |
| 0.2 | 0.9176 | 0.8204 | 0.0972 | 0.265012 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
