# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-15T09:16:54.480873

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.9456 | 0.8824 | 0.0631 | 0.259024 | 148.702133 |
| Original (TFLite) | 0.9335 | 0.8914 | 0.0420 | 0.259024 | 148.702133 |
| QAT+Prune only | 0.6242 | 0.4358 | 0.1885 | 0.259024 | 148.702133 |
| QAT+PTQ | 0.6239 | 0.4369 | 0.1870 | 0.259024 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.259024 | 148.702133 |
| Compressed (PTQ) | 0.6239 | 0.4369 | 0.1870 | 0.259024 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.9474 | 0.8806 | 0.0668 | 0.261320 |
| 0.05 | 0.9474 | 0.8804 | 0.0670 | 0.261645 |
| 0.1 | 0.9474 | 0.8802 | 0.0672 | 0.262097 |
| 0.15 | 0.9474 | 0.8802 | 0.0672 | 0.262594 |
| 0.2 | 0.9474 | 0.8802 | 0.0672 | 0.263158 |

## Epsilon tuning

- **Best epsilon:** 0.1000
- **Target success rate:** 0.50
