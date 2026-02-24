# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-18T01:02:52.082245

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4511 | 0.1835 | 0.2676 | 0.270626 | 148.702133 |
| Original (TFLite) | 0.8786 | 0.5538 | 0.3248 | 0.270626 | 148.702133 |
| QAT+Prune only | 0.8938 | 0.8676 | 0.0262 | 0.270626 | 148.702133 |
| QAT+PTQ | 0.8934 | 0.8678 | 0.0256 | 0.270626 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.270626 | 148.702133 |
| Compressed (PTQ) | 0.8934 | 0.8678 | 0.0256 | 0.270626 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4536 | 0.1846 | 0.2690 | 0.262240 |
| 0.05 | 0.4536 | 0.1846 | 0.2690 | 0.267116 |
| 0.1 | 0.4536 | 0.1846 | 0.2690 | 0.273663 |
| 0.15 | 0.4536 | 0.1846 | 0.2690 | 0.280385 |
| 0.2 | 0.4536 | 0.1846 | 0.2690 | 0.287728 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
