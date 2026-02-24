# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-14T07:29:28.121289

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4624 | 0.2283 | 0.2341 | 0.270323 | 148.702133 |
| Original (TFLite) | 0.5121 | 0.3795 | 0.1326 | 0.270323 | 148.702133 |
| QAT+Prune only | 0.1784 | 0.1762 | 0.0022 | 0.270323 | 148.702133 |
| QAT+PTQ | 0.1784 | 0.1762 | 0.0022 | 0.270323 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.270323 | 148.702133 |
| Compressed (PTQ) | 0.1784 | 0.1762 | 0.0022 | 0.270323 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4714 | 0.2404 | 0.2310 | 0.262192 |
| 0.05 | 0.4714 | 0.2356 | 0.2358 | 0.266918 |
| 0.1 | 0.4714 | 0.2350 | 0.2364 | 0.273234 |
| 0.15 | 0.4714 | 0.2350 | 0.2364 | 0.279824 |
| 0.2 | 0.4714 | 0.2350 | 0.2364 | 0.286833 |

## Epsilon tuning

- **Best epsilon:** 0.1000
- **Target success rate:** 0.50
