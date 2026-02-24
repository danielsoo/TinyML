# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-22T04:06:38.139660

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9194 | 0.9009 | 0.0186 | 0.259658 | 148.702133 |
| Original (TFLite) | 0.8783 | 0.8773 | 0.0010 | 0.259658 | 148.702133 |
| QAT+Prune only | 0.8140 | 0.7955 | 0.0185 | 0.259658 | 148.702133 |
| QAT+PTQ | 0.8141 | 0.7949 | 0.0192 | 0.259658 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259658 | 148.702133 |
| Compressed (PTQ) | 0.8141 | 0.7949 | 0.0192 | 0.259658 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9190 | 0.9066 | 0.0124 | 0.261381 |
| 0.05 | 0.9190 | 0.9048 | 0.0142 | 0.261982 |
| 0.1 | 0.9190 | 0.9032 | 0.0158 | 0.262771 |
| 0.15 | 0.9190 | 0.9022 | 0.0168 | 0.263630 |
| 0.2 | 0.9190 | 0.9002 | 0.0188 | 0.264575 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
