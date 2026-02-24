# FGSM Attack Report

- **Model:** `models/global_model.h5`
- **Dataset:** cicids2017
- **Data path:** data/raw/CIC-IDS2017
- **Max samples:** 2000000
- **Prediction threshold:** 0.3
- **Generated:** 2026-02-10T18:02:38.076673

## Epsilon sweep

| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |
|---------|--------------|---------|--------------|-------------|
| 0.01 | 0.3728 | 0.1802 | 0.1926 | 0.258698 |
| 0.05 | 0.3728 | 0.1802 | 0.1926 | 0.266950 |
| 0.10 | 0.3728 | 0.1802 | 0.1926 | 0.278541 |
| 0.15 | 0.3728 | 0.1802 | 0.1926 | 0.291075 |
| 0.20 | 0.3728 | 0.1802 | 0.1926 | 0.305418 |

## Epsilon tuning

- **Best epsilon:** 0.0100
- **Target success rate:** 0.50

## Final evaluation (adversarial dataset)

- **Epsilon used:** 0.1
- **Original Accuracy:** 0.3774 (37.74%)
- **Adversarial Accuracy:** 0.1784 (17.84%)
- **Attack Success Rate:** 0.1990 (19.90%)
- **Attack Success Count:** 3980/20000
- **Average Perturbation:** 0.278621
- **Max Perturbation:** 211.626068
