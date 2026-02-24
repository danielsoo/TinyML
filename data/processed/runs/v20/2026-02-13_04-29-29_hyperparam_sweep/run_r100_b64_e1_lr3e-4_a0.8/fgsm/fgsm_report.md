# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-19T10:49:33.933216

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4118 | 0.2556 | 0.1562 | 0.271281 | 148.702133 |
| Original (TFLite) | 0.8244 | 0.4099 | 0.4145 | 0.271281 | 148.702133 |
| QAT+Prune only | 0.8599 | 0.8929 | -0.0329 | 0.271281 | 148.702133 |
| QAT+PTQ | 0.8598 | 0.8931 | -0.0333 | 0.271281 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.271281 | 148.702133 |
| Compressed (PTQ) | 0.8598 | 0.8931 | -0.0333 | 0.271281 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4128 | 0.2600 | 0.1528 | 0.262183 |
| 0.05 | 0.4128 | 0.2596 | 0.1532 | 0.267368 |
| 0.1 | 0.4128 | 0.2596 | 0.1532 | 0.274359 |
| 0.15 | 0.4128 | 0.2594 | 0.1534 | 0.281727 |
| 0.2 | 0.4128 | 0.2594 | 0.1534 | 0.289957 |

## Epsilon tuning

- **Best epsilon:** 0.1500
- **Target success rate:** 0.50
