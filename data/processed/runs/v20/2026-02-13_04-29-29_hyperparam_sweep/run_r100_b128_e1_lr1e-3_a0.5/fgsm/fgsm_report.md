# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-21T05:46:26.768949

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4339 | 0.4864 | -0.0525 | 0.267852 | 148.702133 |
| Original (TFLite) | 0.6852 | 0.6299 | 0.0553 | 0.267852 | 148.702133 |
| QAT+Prune only | 0.8590 | 0.8501 | 0.0089 | 0.267852 | 148.702133 |
| QAT+PTQ | 0.8579 | 0.8496 | 0.0083 | 0.267852 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.267852 | 148.702133 |
| Compressed (PTQ) | 0.8579 | 0.8496 | 0.0083 | 0.267852 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4356 | 0.5162 | -0.0806 | 0.261888 |
| 0.05 | 0.4356 | 0.5188 | -0.0832 | 0.265673 |
| 0.1 | 0.4356 | 0.4868 | -0.0512 | 0.271014 |
| 0.15 | 0.4356 | 0.4612 | -0.0256 | 0.276476 |
| 0.2 | 0.4356 | 0.4392 | -0.0036 | 0.282384 |

## Epsilon tuning

- **Best epsilon:** 0.2000
- **Target success rate:** 0.50
