# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-12T14:25:45.338736

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.3724 | 0.1784 | 0.1939 | 0.271837 | 148.702133 |
| Original (TFLite) | 0.9143 | 0.3777 | 0.5365 | 0.271837 | 148.702133 |
| QAT+Prune only | 0.8135 | 0.8717 | -0.0582 | 0.271837 | 148.702133 |
| QAT+PTQ | 0.8139 | 0.8725 | -0.0585 | 0.271837 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.271837 | 148.702133 |
| Compressed (PTQ) | 0.8139 | 0.8725 | -0.0585 | 0.271837 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.3804 | 0.1802 | 0.2002 | 0.262275 |
| 0.05 | 0.3804 | 0.1802 | 0.2002 | 0.267620 |
| 0.1 | 0.3804 | 0.1802 | 0.2002 | 0.274815 |
| 0.15 | 0.3804 | 0.1802 | 0.2002 | 0.282085 |
| 0.2 | 0.3804 | 0.1802 | 0.2002 | 0.289999 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
