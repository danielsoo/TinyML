# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-24T21:22:22.303928

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.8888 | 0.7894 | 0.0993 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.1787 | 0.1784 | 0.0003 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.1797 | 0.1785 | 0.0011 | 0.258160 | 148.702133 |
| saved_model_traditional_qat | 0.9669 | 0.2006 | 0.7664 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.9636 | 0.8822 | 0.0814 | 0.258160 | 148.702133 |
| Compressed (QAT) | 0.9614 | 0.9338 | 0.0276 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8890 | 0.7876 | 0.1014 | 0.261266 |
| 0.05 | 0.8890 | 0.7876 | 0.1014 | 0.261266 |
| 0.1 | 0.8890 | 0.7876 | 0.1014 | 0.261266 |
| 0.15 | 0.8890 | 0.7876 | 0.1014 | 0.261266 |
| 0.2 | 0.8890 | 0.7876 | 0.1014 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
