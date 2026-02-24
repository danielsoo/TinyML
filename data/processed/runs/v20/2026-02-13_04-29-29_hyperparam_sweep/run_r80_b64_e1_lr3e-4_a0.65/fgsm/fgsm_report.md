# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-16T01:53:15.811087

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8216 | 0.8216 | 0.0000 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.8871 | 0.6212 | 0.2659 | 0.258160 | 148.702133 |
| QAT+Prune only | 0.7773 | 0.7448 | 0.0326 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.7766 | 0.7447 | 0.0319 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.258160 | 148.702133 |
| Compressed (PTQ) | 0.7766 | 0.7447 | 0.0319 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.05 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.1 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.15 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |
| 0.2 | 0.8198 | 0.8198 | 0.0000 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
