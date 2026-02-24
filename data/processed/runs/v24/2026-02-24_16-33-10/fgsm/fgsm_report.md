# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-24T16:35:09.157281

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9075 | 0.8534 | 0.0541 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.1829 | 0.1784 | 0.0044 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.1784 | 0.1784 | 0.0000 | 0.258160 | 148.702133 |
| saved_model_traditional_qat | 0.9563 | 0.2576 | 0.6986 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.9394 | 0.9203 | 0.0192 | 0.258160 | 148.702133 |
| Compressed (QAT) | 0.9660 | 0.9293 | 0.0367 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9100 | 0.8536 | 0.0564 | 0.261266 |
| 0.05 | 0.9100 | 0.8536 | 0.0564 | 0.261266 |
| 0.1 | 0.9100 | 0.8536 | 0.0564 | 0.261266 |
| 0.15 | 0.9100 | 0.8536 | 0.0564 | 0.261266 |
| 0.2 | 0.9100 | 0.8536 | 0.0564 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
