# FGSM Attack Report

- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Adversarial examples generated with:** `models/global_model.h5`
- **Epsilon used:** 0.1
- **Generated:** 2026-02-22T10:41:07.756232

## Model comparison (same adversarial examples)

| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |
|-------|--------------|---------|--------------|-------------|-------------|
| Keras (global_model.h5) | 0.4703 | 0.1911 | 0.2792 | 0.272014 | 148.702133 |
| Original (TFLite) | 0.8422 | 0.6330 | 0.2092 | 0.272014 | 148.702133 |
| QAT+Prune only | 0.3180 | 0.5843 | -0.2663 | 0.272014 | 148.702133 |
| QAT+PTQ | 0.3175 | 0.5819 | -0.2645 | 0.272014 | 148.702133 |
| noQAT+PTQ | 0.8216 | 0.8216 | 0.0000 | 0.272014 | 148.702133 |
| Compressed (PTQ) | 0.3175 | 0.5819 | -0.2645 | 0.272014 | 148.702133 |

## Epsilon sweep (attack model)

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.4772 | 0.1926 | 0.2846 | 0.262261 |
| 0.05 | 0.4772 | 0.1926 | 0.2846 | 0.267658 |
| 0.1 | 0.4772 | 0.1926 | 0.2846 | 0.274939 |
| 0.15 | 0.4772 | 0.1926 | 0.2846 | 0.282897 |
| 0.2 | 0.4772 | 0.1926 | 0.2846 | 0.291960 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50
