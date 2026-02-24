# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-11T07:15:50.949481

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8190 | 0.8190 | 0.0000 | 0.257536 | 81.822029 |
| Original (TFLite) | 0.4899 | 0.4451 | 0.0448 | 0.257536 | 81.822029 |
| QAT+Prune only | 0.9493 | 0.9125 | 0.0367 | 0.257536 | 81.822029 |
| QAT+PTQ | 0.9495 | 0.9120 | 0.0374 | 0.257536 | 81.822029 |
| noQAT+PTQ | 0.8190 | 0.8190 | 0.0000 | 0.257536 | 81.822029 |
| Compressed (PTQ) | 0.9495 | 0.9120 | 0.0374 | 0.257536 | 81.822029 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8156 | 0.8156 | 0.0000 | 0.258051 |
| 0.05 | 0.8156 | 0.8156 | 0.0000 | 0.258051 |
| 0.1 | 0.8156 | 0.8156 | 0.0000 | 0.258051 |
| 0.15 | 0.8156 | 0.8156 | 0.0000 | 0.258051 |
| 0.2 | 0.8156 | 0.8156 | 0.0000 | 0.258051 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
