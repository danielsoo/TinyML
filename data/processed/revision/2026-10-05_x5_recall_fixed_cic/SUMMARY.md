# Revision experiments (full)

- commit: 6d2d66918a181b798bd83b548de0d495c54b9783
- configs: config/jobs/2026-10-05_x5_recall_fixed_cic (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- recall_priority: 2 min

# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cic_near_iid | fp32_federated | fixed 0.1 | 1 | 802.44 | 0.1 | 11 | 11 | 99.987 | 22.23 | 22.23 | 64.88 |
| cic_near_iid | fp32_federated | fixed 0.2 | 1 | 802.44 | 0.2 | 22 | 22 | 99.974 | 11.01 | 11.01 | 78.85 |
| cic_near_iid | fp32_federated | fixed 0.3 | 1 | 802.44 | 0.3 | 52 | 52 | 99.939 | 6.84 | 6.84 | 85.7 |
| cic_near_iid | fp32_federated | val recall>=0.999 | 1 | 802.44 | 0.3276 | 103 | 103 | 99.879 | 6.3 | 6.3 | 86.64 |
| cic_near_iid | fp32_federated | val recall>=0.9995 | 1 | 802.44 | 0.3147 | 67 | 67 | 99.921 | 6.58 | 6.58 | 86.16 |
| cic_near_iid | fp32_federated | val recall>=0.9999 | 1 | 802.44 | 0.2692 | 37 | 37 | 99.957 | 7.56 | 7.56 | 84.43 |
| cic_near_iid | kdclip_p30 | fixed 0.1 | 3 | 112.06 | 0.1 | 3.7 | 6 | 99.996 | 34.8 | 35.53 | 54.13 |
| cic_near_iid | kdclip_p30 | fixed 0.2 | 3 | 112.06 | 0.2 | 32.7 | 33 | 99.962 | 8.47 | 8.83 | 82.89 |
| cic_near_iid | kdclip_p30 | fixed 0.3 | 3 | 112.06 | 0.3 | 186 | 231 | 99.782 | 5.15 | 5.25 | 88.74 |
| cic_near_iid | kdclip_p30 | val recall>=0.999 | 3 | 112.06 | 0.2539 | 106 | 138 | 99.876 | 6.27 | 6.76 | 86.7 |
| cic_near_iid | kdclip_p30 | val recall>=0.9995 | 3 | 112.06 | 0.2214 | 53.7 | 73 | 99.937 | 7.63 | 8.37 | 84.33 |
| cic_near_iid | kdclip_p30 | val recall>=0.9999 | 3 | 112.06 | 0.1888 | 23.3 | 27 | 99.973 | 9.65 | 10.02 | 80.97 |
| cic_near_iid | kdclip_p30_fix | fixed 0.1 | 3 | 112.07 | 0.1 | 4.7 | 7 | 99.995 | 30.57 | 37.68 | 57.6 |
| cic_near_iid | kdclip_p30_fix | fixed 0.2 | 3 | 112.07 | 0.2 | 35.3 | 42 | 99.959 | 7.99 | 9.03 | 83.71 |
| cic_near_iid | kdclip_p30_fix | fixed 0.3 | 3 | 112.07 | 0.3 | 183 | 219 | 99.785 | 5.01 | 5.26 | 89.03 |
| cic_near_iid | kdclip_p30_fix | val recall>=0.999 | 3 | 112.07 | 0.2448 | 117 | 137 | 99.863 | 6.03 | 6.52 | 87.14 |
| cic_near_iid | kdclip_p30_fix | val recall>=0.9995 | 3 | 112.07 | 0.2135 | 52 | 74 | 99.939 | 7.47 | 8.56 | 84.61 |
| cic_near_iid | kdclip_p30_fix | val recall>=0.9999 | 3 | 112.07 | 0.1953 | 28.7 | 33 | 99.966 | 8.63 | 10.31 | 82.68 |
| cic_near_iid | kdclip_p30_fixmax | fixed 0.1 | 3 | 112.07 | 0.1 | 7 | 9 | 99.992 | 32.28 | 38.54 | 56.19 |
| cic_near_iid | kdclip_p30_fixmax | fixed 0.2 | 3 | 112.07 | 0.2 | 32.7 | 36 | 99.962 | 8.16 | 9.01 | 83.43 |
| cic_near_iid | kdclip_p30_fixmax | fixed 0.3 | 3 | 112.07 | 0.3 | 188.3 | 220 | 99.779 | 4.97 | 5.19 | 89.11 |
| cic_near_iid | kdclip_p30_fixmax | val recall>=0.999 | 3 | 112.07 | 0.237 | 95.3 | 134 | 99.888 | 6.54 | 7.23 | 86.24 |
| cic_near_iid | kdclip_p30_fixmax | val recall>=0.9995 | 3 | 112.07 | 0.2161 | 53.7 | 84 | 99.937 | 7.61 | 9.01 | 84.39 |
| cic_near_iid | kdclip_p30_fixmax | val recall>=0.9999 | 3 | 112.07 | 0.194 | 26.3 | 34 | 99.969 | 9.33 | 12.27 | 81.62 |
| cic_near_iid | kdclip_p30_cle | fixed 0.1 | 3 | 112.07 | 0.1 | 14 | 20 | 99.984 | 34.01 | 37.4 | 54.8 |
| cic_near_iid | kdclip_p30_cle | fixed 0.2 | 3 | 112.07 | 0.2 | 52.7 | 59 | 99.938 | 9.18 | 9.64 | 81.72 |
| cic_near_iid | kdclip_p30_cle | fixed 0.3 | 3 | 112.07 | 0.3 | 300.7 | 456 | 99.647 | 5.49 | 5.9 | 88.04 |
| cic_near_iid | kdclip_p30_cle | val recall>=0.999 | 3 | 112.07 | 0.2357 | 137 | 187 | 99.839 | 7.34 | 8.85 | 84.83 |
| cic_near_iid | kdclip_p30_cle | val recall>=0.9995 | 3 | 112.07 | 0.2109 | 75.7 | 109 | 99.911 | 8.82 | 11.18 | 82.39 |
| cic_near_iid | kdclip_p30_cle | val recall>=0.9999 | 3 | 112.07 | 0.1471 | 29.3 | 43 | 99.966 | 35.27 | 84.02 | 63.65 |
| cic_near_iid | kdclip_p30_fix_cle | fixed 0.1 | 3 | 112.07 | 0.1 | 7.7 | 10 | 99.991 | 32.72 | 37.32 | 55.77 |
| cic_near_iid | kdclip_p30_fix_cle | fixed 0.2 | 3 | 112.07 | 0.2 | 68 | 88 | 99.92 | 8.78 | 8.98 | 82.36 |
| cic_near_iid | kdclip_p30_fix_cle | fixed 0.3 | 3 | 112.07 | 0.3 | 214 | 227 | 99.749 | 5.62 | 5.83 | 87.83 |
| cic_near_iid | kdclip_p30_fix_cle | val recall>=0.999 | 3 | 112.07 | 0.2305 | 109.7 | 155 | 99.871 | 7.51 | 9.04 | 84.53 |
| cic_near_iid | kdclip_p30_fix_cle | val recall>=0.9995 | 3 | 112.07 | 0.2083 | 77 | 103 | 99.91 | 8.9 | 11.41 | 82.28 |
| cic_near_iid | kdclip_p30_fix_cle | val recall>=0.9999 | 3 | 112.07 | 0.194 | 36.3 | 59 | 99.957 | 9.97 | 11.86 | 80.56 |

