# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-22T07:23:34.765177

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9339 | 0.8216 | 0.1124 | 0.259298 | 148.702133 |
| Original (TFLite) | 0.9109 | 0.9012 | 0.0097 | 0.259298 | 148.702133 |
| QAT+Prune only | 0.6726 | 0.6967 | -0.0241 | 0.259298 | 148.702133 |
| QAT+PTQ | 0.6705 | 0.6963 | -0.0257 | 0.259298 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259298 | 148.702133 |
| Compressed (PTQ) | 0.6705 | 0.6963 | -0.0257 | 0.259298 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9386 | 0.8198 | 0.1188 | 0.261330 |
| 0.05 | 0.9386 | 0.8198 | 0.1188 | 0.261735 |
| 0.1 | 0.9386 | 0.8196 | 0.1190 | 0.262309 |
| 0.15 | 0.9386 | 0.8196 | 0.1190 | 0.262954 |
| 0.2 | 0.9386 | 0.8196 | 0.1190 | 0.263697 |

## Epsilon tuning

- **Best epsilon:** 0.1000
- **Target success rate:** 0.50
