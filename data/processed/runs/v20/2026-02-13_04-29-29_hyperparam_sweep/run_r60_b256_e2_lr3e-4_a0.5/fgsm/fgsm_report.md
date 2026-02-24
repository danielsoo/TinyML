# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T13:24:45.707551

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9278 | 0.8798 | 0.0480 | 0.259460 | 148.702133 |
| Original (TFLite) | 0.9012 | 0.8912 | 0.0100 | 0.259460 | 148.702133 |
| QAT+Prune only | 0.8991 | 0.8606 | 0.0386 | 0.259460 | 148.702133 |
| QAT+PTQ | 0.8999 | 0.8606 | 0.0394 | 0.259460 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259460 | 148.702133 |
| Compressed (PTQ) | 0.8999 | 0.8606 | 0.0394 | 0.259460 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9298 | 0.8784 | 0.0514 | 0.261363 |
| 0.05 | 0.9298 | 0.8784 | 0.0514 | 0.261856 |
| 0.1 | 0.9298 | 0.8784 | 0.0514 | 0.262538 |
| 0.15 | 0.9298 | 0.8782 | 0.0516 | 0.263250 |
| 0.2 | 0.9298 | 0.8772 | 0.0526 | 0.264029 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
