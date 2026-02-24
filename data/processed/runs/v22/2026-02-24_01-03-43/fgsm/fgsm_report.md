# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-24T01:05:45.443830

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9093 | 0.8632 | 0.0461 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.1722 | 0.1784 | -0.0062 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.2194 | 0.1789 | 0.0406 | 0.258160 | 148.702133 |
| saved_model_traditional_qat | 0.9639 | 0.6534 | 0.3105 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.1790 | 0.1784 | 0.0006 | 0.258160 | 148.702133 |
| Compressed (QAT) | 0.9603 | 0.9189 | 0.0414 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9118 | 0.8648 | 0.0470 | 0.261266 |
| 0.05 | 0.9118 | 0.8648 | 0.0470 | 0.261266 |
| 0.1 | 0.9118 | 0.8648 | 0.0470 | 0.261266 |
| 0.15 | 0.9118 | 0.8648 | 0.0470 | 0.261266 |
| 0.2 | 0.9118 | 0.8648 | 0.0470 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
