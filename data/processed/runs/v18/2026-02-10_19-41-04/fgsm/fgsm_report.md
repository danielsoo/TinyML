# FGSM Attack Report

- **Model:** `models/global_model.h5`
- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Generated:** 2026-02-10T19:43:54.253868

## Epsilon sweep

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.8274 | 0.8198 | 0.0076 | 0.258708 |
| 0.05 | 0.8274 | 0.8198 | 0.0076 | 0.267535 |
| 0.10 | 0.8274 | 0.8198 | 0.0076 | 0.279116 |
| 0.15 | 0.8274 | 0.8198 | 0.0076 | 0.291589 |
| 0.20 | 0.8274 | 0.8198 | 0.0076 | 0.304895 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50

## Final evaluation (adversarial dataset)

- **Epsilon used:** 0.1
- **Original Accuracy:** 0.8319 (83.19%)
- **Adversarial Accuracy:** 0.8216 (82.16%)
- **Attack Success Rate:** 0.0103 (1.03%)
- **Attack Success Count:** 206/20000
- **Average Perturbation:** 0.279028
- **Max Perturbation:** 211.626068
