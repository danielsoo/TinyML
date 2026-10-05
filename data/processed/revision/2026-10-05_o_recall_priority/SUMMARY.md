# Revision experiments (full)

- commit: bbd2dfb12556d6721d7890488486a2cee5b92160
- configs: config/jobs/2026-10-05_o_recall_priority (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- recall_priority: 3 min

# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cic_near_iid | fp32_federated | fixed 0.05 | 1 | 802.44 | 0.05 | 7 | 7 | 99.992 | 37.47 | 37.47 | 52.29 |
| cic_near_iid | fp32_federated | fixed 0.1 | 1 | 802.44 | 0.1 | 11 | 11 | 99.987 | 22.23 | 22.23 | 64.88 |
| cic_near_iid | fp32_federated | fixed 0.2 | 1 | 802.44 | 0.2 | 22 | 22 | 99.974 | 11.01 | 11.01 | 78.85 |
| cic_near_iid | fp32_federated | fixed 0.3 | 1 | 802.44 | 0.3 | 52 | 52 | 99.939 | 6.84 | 6.84 | 85.7 |
| cic_near_iid | fp32_federated | fixed 0.5 | 1 | 802.44 | 0.5 | 4499 | 4499 | 94.718 | 2.95 | 2.95 | 90.61 |
| cic_near_iid | fp32_federated | val recall>=0.999 | 1 | 802.44 | 0.3276 | 103 | 103 | 99.879 | 6.3 | 6.3 | 86.64 |
| cic_near_iid | fp32_federated | val recall>=0.9995 | 1 | 802.44 | 0.3147 | 67 | 67 | 99.921 | 6.58 | 6.58 | 86.16 |
| cic_near_iid | fp32_federated | val recall>=0.9999 | 1 | 802.44 | 0.2692 | 37 | 37 | 99.957 | 7.56 | 7.56 | 84.43 |
| cic_near_iid | hard | fixed 0.05 | 3 | 65.43 | 0.05 | 251.7 | 296 | 99.705 | 8.82 | 9.05 | 82.17 |
| cic_near_iid | hard | fixed 0.1 | 3 | 65.43 | 0.1 | 304.7 | 337 | 99.642 | 7.61 | 7.75 | 84.2 |
| cic_near_iid | hard | fixed 0.2 | 3 | 65.43 | 0.2 | 403 | 464 | 99.527 | 6.44 | 6.7 | 86.22 |
| cic_near_iid | hard | fixed 0.3 | 3 | 65.43 | 0.3 | 462.7 | 483 | 99.457 | 5.97 | 6.38 | 87.04 |
| cic_near_iid | hard | fixed 0.5 | 3 | 65.43 | 0.5 | 1516 | 2729 | 98.22 | 4.57 | 5.74 | 89.12 |
| cic_near_iid | hard | val recall>=0.999 | 3 | 65.43 | 0.0117 | 86.3 | 141 | 99.899 | 15.78 | 17.48 | 72.24 |
| cic_near_iid | hard | val recall>=0.9995 | 3 | 65.43 | 0.0052 | 44.3 | 58 | 99.948 | 23.39 | 27.87 | 63.88 |
| cic_near_iid | hard | val recall>=0.9999 | 3 | 65.43 | 0.0039 | 41.7 | 58 | 99.951 | 25.27 | 27.87 | 61.94 |
| cic_near_iid | kd05 | fixed 0.05 | 3 | 65.44 | 0.05 | 0 | 0 | 100.0 | 95.83 | 96.84 | 30.0 |
| cic_near_iid | kd05 | fixed 0.1 | 3 | 65.44 | 0.1 | 8 | 13 | 99.991 | 39.35 | 47.6 | 51.35 |
| cic_near_iid | kd05 | fixed 0.2 | 3 | 65.44 | 0.2 | 151.7 | 204 | 99.822 | 12.97 | 15.75 | 76.02 |
| cic_near_iid | kd05 | fixed 0.3 | 3 | 65.44 | 0.3 | 322.7 | 333 | 99.621 | 7.48 | 7.76 | 84.42 |
| cic_near_iid | kd05 | fixed 0.5 | 3 | 65.44 | 0.5 | 1064.3 | 1231 | 98.75 | 5.09 | 5.61 | 88.37 |
| cic_near_iid | kd05 | val recall>=0.999 | 3 | 65.44 | 0.1823 | 86.7 | 114 | 99.898 | 16.42 | 16.95 | 71.41 |
| cic_near_iid | kd05 | val recall>=0.9995 | 3 | 65.44 | 0.1641 | 37.3 | 52 | 99.956 | 19.92 | 24.25 | 67.5 |
| cic_near_iid | kd05 | val recall>=0.9999 | 3 | 65.44 | 0.1562 | 28.3 | 40 | 99.967 | 21.3 | 24.25 | 65.91 |
| cic_near_iid | kd0 | fixed 0.05 | 3 | 65.44 | 0.05 | 0 | 0 | 100.0 | 99.87 | 99.98 | 29.14 |
| cic_near_iid | kd0 | fixed 0.1 | 3 | 65.44 | 0.1 | 0.3 | 1 | 100.0 | 95.98 | 96.47 | 29.97 |
| cic_near_iid | kd0 | fixed 0.2 | 3 | 65.44 | 0.2 | 14.3 | 18 | 99.983 | 33.18 | 35.01 | 55.33 |
| cic_near_iid | kd0 | fixed 0.3 | 3 | 65.44 | 0.3 | 90.7 | 133 | 99.894 | 18.18 | 19.47 | 69.29 |
| cic_near_iid | kd0 | fixed 0.5 | 3 | 65.44 | 0.5 | 1400.7 | 1560 | 98.356 | 5.2 | 5.29 | 87.95 |
| cic_near_iid | kd0 | val recall>=0.999 | 3 | 65.44 | 0.3177 | 81 | 92 | 99.905 | 17.11 | 19.21 | 70.61 |
| cic_near_iid | kd0 | val recall>=0.9995 | 3 | 65.44 | 0.3021 | 66.3 | 80 | 99.922 | 18.53 | 21.23 | 69.02 |
| cic_near_iid | kd0 | val recall>=0.9999 | 3 | 65.44 | 0.2148 | 15.7 | 27 | 99.982 | 35.95 | 51.08 | 54.27 |
| cic_near_iid | cw4 | fixed 0.05 | 3 | 65.44 | 0.05 | 126.7 | 137 | 99.851 | 13.49 | 15.66 | 75.3 |
| cic_near_iid | cw4 | fixed 0.1 | 3 | 65.44 | 0.1 | 187.7 | 226 | 99.78 | 10.5 | 12.89 | 79.63 |
| cic_near_iid | cw4 | fixed 0.2 | 3 | 65.44 | 0.2 | 275.7 | 309 | 99.676 | 8.02 | 8.55 | 83.52 |
| cic_near_iid | cw4 | fixed 0.3 | 3 | 65.44 | 0.3 | 316.7 | 332 | 99.628 | 7.24 | 7.62 | 84.84 |
| cic_near_iid | cw4 | fixed 0.5 | 3 | 65.44 | 0.5 | 404 | 438 | 99.526 | 6.34 | 6.78 | 86.4 |
| cic_near_iid | cw4 | val recall>=0.999 | 3 | 65.44 | 0.0664 | 101.3 | 142 | 99.881 | 16.2 | 22.76 | 72.11 |
| cic_near_iid | cw4 | val recall>=0.9995 | 3 | 65.44 | 0.0273 | 63 | 81 | 99.926 | 20.31 | 26.37 | 67.2 |
| cic_near_iid | cw4 | val recall>=0.9999 | 3 | 65.44 | 0.0078 | 39 | 55 | 99.954 | 28.23 | 35.18 | 59.54 |
| cic_near_iid | kd05_clip | fixed 0.05 | 3 | 65.44 | 0.05 | 1 | 2 | 99.999 | 97.02 | 98.64 | 29.75 |
| cic_near_iid | kd05_clip | fixed 0.1 | 3 | 65.44 | 0.1 | 7 | 12 | 99.992 | 34.35 | 39.06 | 54.63 |
| cic_near_iid | kd05_clip | fixed 0.2 | 3 | 65.44 | 0.2 | 61 | 101 | 99.928 | 9.92 | 10.88 | 80.52 |
| cic_near_iid | kd05_clip | fixed 0.3 | 3 | 65.44 | 0.3 | 231.3 | 292 | 99.728 | 6.04 | 6.14 | 87.04 |
| cic_near_iid | kd05_clip | fixed 0.5 | 3 | 65.44 | 0.5 | 534.3 | 594 | 99.373 | 4.65 | 5.02 | 89.52 |
| cic_near_iid | kd05_clip | val recall>=0.999 | 3 | 65.44 | 0.2135 | 77 | 133 | 99.91 | 9.54 | 11.62 | 81.23 |
| cic_near_iid | kd05_clip | val recall>=0.9995 | 3 | 65.44 | 0.1979 | 44 | 55 | 99.948 | 10.43 | 11.62 | 79.8 |
| cic_near_iid | kd05_clip | val recall>=0.9999 | 3 | 65.44 | 0.1706 | 23.7 | 25 | 99.972 | 14.33 | 15.84 | 74.15 |
| cic_near_iid | kd05_ft50k | fixed 0.05 | 3 | 65.44 | 0.05 | 4.7 | 7 | 99.995 | 93.5 | 94.02 | 30.52 |
| cic_near_iid | kd05_ft50k | fixed 0.1 | 3 | 65.44 | 0.1 | 12.3 | 14 | 99.986 | 29.4 | 32.08 | 58.32 |
| cic_near_iid | kd05_ft50k | fixed 0.2 | 3 | 65.44 | 0.2 | 286 | 458 | 99.664 | 9.85 | 13.26 | 80.7 |
| cic_near_iid | kd05_ft50k | fixed 0.3 | 3 | 65.44 | 0.3 | 536.7 | 969 | 99.37 | 7.44 | 10.21 | 84.5 |
| cic_near_iid | kd05_ft50k | fixed 0.5 | 3 | 65.44 | 0.5 | 1258 | 1993 | 98.523 | 5.83 | 7.94 | 86.92 |
| cic_near_iid | kd05_ft50k | val recall>=0.999 | 3 | 65.44 | 0.1654 | 73 | 115 | 99.914 | 15.55 | 24.52 | 73.35 |
| cic_near_iid | kd05_ft50k | val recall>=0.9995 | 3 | 65.44 | 0.1589 | 39.7 | 56 | 99.953 | 17.94 | 24.52 | 70.09 |
| cic_near_iid | kd05_ft50k | val recall>=0.9999 | 3 | 65.44 | 0.138 | 26 | 28 | 99.969 | 20.28 | 27.37 | 67.36 |

