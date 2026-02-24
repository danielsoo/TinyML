# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T10:06:32.516137

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.3392 | 0.1784 | 0.1607 | 0.272788 | 148.702133 |
| Original (TFLite) | 0.6727 | 0.4783 | 0.1943 | 0.272788 | 148.702133 |
| QAT+Prune only | 0.8360 | 0.8031 | 0.0330 | 0.272788 | 148.702133 |
| QAT+PTQ | 0.8348 | 0.8024 | 0.0323 | 0.272788 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.272788 | 148.702133 |
| Compressed (PTQ) | 0.8348 | 0.8024 | 0.0323 | 0.272788 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.3446 | 0.1802 | 0.1644 | 0.262397 |
| 0.05 | 0.3446 | 0.1802 | 0.1644 | 0.268186 |
| 0.1 | 0.3446 | 0.1802 | 0.1644 | 0.275788 |
| 0.15 | 0.3446 | 0.1802 | 0.1644 | 0.283506 |
| 0.2 | 0.3446 | 0.1802 | 0.1644 | 0.291780 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
