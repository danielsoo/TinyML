# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-23T02:37:04.893219

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.3580 | 0.1865 | 0.1716 | 0.272099 | 148.702133 |
| Original (TFLite) | 0.6350 | 0.4741 | 0.1609 | 0.272099 | 148.702133 |
| QAT+Prune only | 0.7723 | 0.7007 | 0.0716 | 0.272099 | 148.702133 |
| QAT+PTQ | 0.7723 | 0.7007 | 0.0716 | 0.272099 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.272099 | 148.702133 |
| Compressed (PTQ) | 0.7723 | 0.7007 | 0.0716 | 0.272099 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.3628 | 0.1884 | 0.1744 | 0.262313 |
| 0.05 | 0.3628 | 0.1884 | 0.1744 | 0.267795 |
| 0.1 | 0.3628 | 0.1884 | 0.1744 | 0.275095 |
| 0.15 | 0.3628 | 0.1884 | 0.1744 | 0.282815 |
| 0.2 | 0.3628 | 0.1884 | 0.1744 | 0.291540 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
