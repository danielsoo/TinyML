# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-18T14:06:57.664231

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8772 | 0.8216 | 0.0556 | 0.261064 | 148.702133 |
| Original (TFLite) | 0.9213 | 0.5336 | 0.3876 | 0.261064 | 148.702133 |
| QAT+Prune only | 0.7430 | 0.6422 | 0.1008 | 0.261064 | 148.702133 |
| QAT+PTQ | 0.7434 | 0.6433 | 0.1001 | 0.261064 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.261064 | 148.702133 |
| Compressed (PTQ) | 0.7434 | 0.6433 | 0.1001 | 0.261064 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8750 | 0.8198 | 0.0552 | 0.261454 |
| 0.05 | 0.8750 | 0.8198 | 0.0552 | 0.262628 |
| 0.1 | 0.8750 | 0.8198 | 0.0552 | 0.264230 |
| 0.15 | 0.8750 | 0.8198 | 0.0552 | 0.266047 |
| 0.2 | 0.8750 | 0.8198 | 0.0552 | 0.268122 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
