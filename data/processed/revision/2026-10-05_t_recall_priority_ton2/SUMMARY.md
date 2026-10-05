# Revision experiments (full)

- commit: d7e8554cc8426dd72a9f6cd4e153611c157d4b0d
- configs: config/jobs/2026-10-05_t_recall_priority_ton2 (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- recall_priority: 1 min

# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton_near_iid_v2 | fp32_federated | fixed 0.1 | 1 | 732.44 | 0.1 | 1 | 1 | 99.994 | 54.57 | 54.57 | 92.49 |
| ton_near_iid_v2 | fp32_federated | fixed 0.2 | 1 | 732.44 | 0.2 | 4 | 4 | 99.975 | 7.82 | 7.82 | 98.84 |
| ton_near_iid_v2 | fp32_federated | fixed 0.3 | 1 | 732.44 | 0.3 | 19 | 19 | 99.883 | 3.61 | 3.61 | 99.41 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.999 | 1 | 732.44 | 0.3165 | 23 | 23 | 99.858 | 3.51 | 3.51 | 99.41 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.9995 | 1 | 732.44 | 0.2591 | 9 | 9 | 99.944 | 4.71 | 4.71 | 99.28 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.9999 | 1 | 732.44 | 0.1105 | 1 | 1 | 99.994 | 23.79 | 23.79 | 96.58 |
| ton_near_iid_v2 | hard | fixed 0.1 | 3 | 56.68 | 0.1 | 138 | 228 | 99.149 | 7.41 | 15.43 | 98.49 |
| ton_near_iid_v2 | hard | fixed 0.2 | 3 | 56.68 | 0.2 | 155.3 | 254 | 99.042 | 6.84 | 14.71 | 98.52 |
| ton_near_iid_v2 | hard | fixed 0.3 | 3 | 56.68 | 0.3 | 168.3 | 274 | 98.962 | 6.45 | 14.58 | 98.53 |
| ton_near_iid_v2 | hard | val recall>=0.999 | 3 | 56.68 | 0.0378 | 15.3 | 16 | 99.905 | 37.65 | 49.41 | 94.7 |
| ton_near_iid_v2 | hard | val recall>=0.9995 | 3 | 56.68 | 0.0078 | 6 | 13 | 99.963 | 57.56 | 100.0 | 92.27 |
| ton_near_iid_v2 | hard | val recall>=0.9999 | 3 | 56.68 | 0.0 | 0 | 0 | 100.0 | 100.0 | 100.0 | 87.06 |
| ton_near_iid_v2 | clipqat | fixed 0.1 | 3 | 56.69 | 0.1 | 37.7 | 57 | 99.768 | 3.53 | 4.17 | 99.36 |
| ton_near_iid_v2 | clipqat | fixed 0.2 | 3 | 56.69 | 0.2 | 58 | 70 | 99.642 | 2.58 | 3.3 | 99.44 |
| ton_near_iid_v2 | clipqat | fixed 0.3 | 3 | 56.69 | 0.3 | 80.3 | 100 | 99.505 | 1.78 | 2.12 | 99.49 |
| ton_near_iid_v2 | clipqat | val recall>=0.999 | 3 | 56.69 | 0.0234 | 13 | 28 | 99.92 | 37.25 | 100.0 | 95.07 |
| ton_near_iid_v2 | clipqat | val recall>=0.9995 | 3 | 56.69 | 0.0065 | 6.3 | 16 | 99.961 | 40.45 | 100.0 | 94.63 |
| ton_near_iid_v2 | clipqat | val recall>=0.9999 | 3 | 56.69 | 0.0 | 0 | 0 | 100.0 | 100.0 | 100.0 | 87.06 |
| ton_near_iid_v2 | kdclip | fixed 0.1 | 3 | 56.69 | 0.1 | 0.7 | 1 | 99.996 | 91.32 | 95.42 | 88.05 |
| ton_near_iid_v2 | kdclip | fixed 0.2 | 3 | 56.69 | 0.2 | 17.3 | 25 | 99.893 | 6.74 | 7.38 | 98.95 |
| ton_near_iid_v2 | kdclip | fixed 0.3 | 3 | 56.69 | 0.3 | 51.7 | 64 | 99.681 | 2.68 | 3.13 | 99.44 |
| ton_near_iid_v2 | kdclip | val recall>=0.999 | 3 | 56.69 | 0.2487 | 21 | 29 | 99.87 | 4.91 | 6.33 | 99.21 |
| ton_near_iid_v2 | kdclip | val recall>=0.9995 | 3 | 56.69 | 0.1992 | 14 | 26 | 99.914 | 9.34 | 16.35 | 98.59 |
| ton_near_iid_v2 | kdclip | val recall>=0.9999 | 3 | 56.69 | 0.1654 | 3.7 | 8 | 99.977 | 12.3 | 17.32 | 98.2 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.1 | 3 | 56.69 | 0.1 | 0.7 | 1 | 99.996 | 85.38 | 85.77 | 88.74 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.2 | 3 | 56.69 | 0.2 | 14.3 | 17 | 99.912 | 6.31 | 6.51 | 99.03 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.3 | 3 | 56.69 | 0.3 | 66 | 68 | 99.593 | 1.74 | 1.91 | 99.54 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.999 | 3 | 56.69 | 0.2435 | 30 | 32 | 99.815 | 3.68 | 3.82 | 99.36 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.9995 | 3 | 56.69 | 0.2148 | 17.7 | 20 | 99.891 | 5.61 | 5.99 | 99.12 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.9999 | 3 | 56.69 | 0.2031 | 12 | 14 | 99.926 | 6.61 | 6.82 | 98.99 |
| ton_near_iid_v2 | kdclip_p30 | fixed 0.1 | 3 | 99.84 | 0.1 | 0 | 0 | 100.0 | 91.91 | 95.64 | 87.98 |
| ton_near_iid_v2 | kdclip_p30 | fixed 0.2 | 3 | 99.84 | 0.2 | 13 | 17 | 99.92 | 6.7 | 7.14 | 98.97 |
| ton_near_iid_v2 | kdclip_p30 | fixed 0.3 | 3 | 99.84 | 0.3 | 49 | 56 | 99.698 | 2.63 | 2.74 | 99.46 |
| ton_near_iid_v2 | kdclip_p30 | val recall>=0.999 | 3 | 99.84 | 0.2513 | 22.7 | 25 | 99.86 | 4.54 | 5.46 | 99.26 |
| ton_near_iid_v2 | kdclip_p30 | val recall>=0.9995 | 3 | 99.84 | 0.2148 | 13 | 19 | 99.92 | 6.17 | 7.05 | 99.05 |
| ton_near_iid_v2 | kdclip_p30 | val recall>=0.9999 | 3 | 99.84 | 0.1862 | 5 | 8 | 99.969 | 9.07 | 12.47 | 98.66 |

