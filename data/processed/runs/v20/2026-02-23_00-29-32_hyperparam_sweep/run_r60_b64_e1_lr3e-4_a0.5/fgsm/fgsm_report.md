# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-23T01:21:40.228229

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.6145 | 0.3937 | 0.2208 | 0.266384 | 148.702133 |
| Original (TFLite) | 0.6059 | 0.6685 | -0.0626 | 0.266384 | 148.702133 |
| QAT+Prune only | 0.6976 | 0.7977 | -0.1002 | 0.266384 | 148.702133 |
| QAT+PTQ | 0.6993 | 0.7975 | -0.0982 | 0.266384 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.266384 | 148.702133 |
| Compressed (PTQ) | 0.6993 | 0.7975 | -0.0982 | 0.266384 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.6248 | 0.4020 | 0.2228 | 0.261773 |
| 0.05 | 0.6248 | 0.4018 | 0.2230 | 0.264909 |
| 0.1 | 0.6248 | 0.4012 | 0.2236 | 0.269254 |
| 0.15 | 0.6248 | 0.4012 | 0.2236 | 0.273738 |
| 0.2 | 0.6248 | 0.4016 | 0.2232 | 0.278467 |

## Epsilon tuning

- **Best epsilon:** 0.1000
- **Target success rate:** 0.50
