# FGSM Attack Report

- **Model:** `models/global_model.h5`
- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Generated:** 2026-02-10T00:19:18.613279

## Epsilon sweep

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8198 | 0.8198 | 0.0000 | nan |
| 0.05 | 0.8198 | 0.8198 | 0.0000 | nan |
| 0.10 | 0.8198 | 0.8198 | 0.0000 | nan |
| 0.15 | 0.8198 | 0.8198 | 0.0000 | nan |
| 0.20 | 0.8198 | 0.8198 | 0.0000 | nan |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50

## Final evaluation (adversarial dataset)

- **Epsilon used:** 0.1
- **Original Accuracy:** 0.8216 (82.16%)
- **Adversarial Accuracy:** 0.8216 (82.16%)
- **Attack Success Rate:** 0.0000 (0.00%)
- **Attack Success Count:** 0/20000
- **Average Perturbation:** nan
- **Max Perturbation:** nan
