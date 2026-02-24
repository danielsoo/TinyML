# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-24T18:40:27.895456

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9071 | 0.8060 | 0.1011 | 0.258160 | 148.702133 |
| Original (TFLite) | 0.1784 | 0.1784 | 0.0000 | 0.258160 | 148.702133 |
| noQAT+PTQ | 0.1784 | 0.1784 | 0.0000 | 0.258160 | 148.702133 |
| saved_model_traditional_qat | 0.9649 | 0.2089 | 0.7560 | 0.258160 | 148.702133 |
| QAT+PTQ | 0.9469 | 0.9170 | 0.0299 | 0.258160 | 148.702133 |
| Compressed (QAT) | 0.9627 | 0.9187 | 0.0439 | 0.258160 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9090 | 0.8076 | 0.1014 | 0.261266 |
| 0.05 | 0.9090 | 0.8076 | 0.1014 | 0.261266 |
| 0.1 | 0.9090 | 0.8076 | 0.1014 | 0.261266 |
| 0.15 | 0.9090 | 0.8076 | 0.1014 | 0.261266 |
| 0.2 | 0.9090 | 0.8076 | 0.1014 | 0.261266 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
