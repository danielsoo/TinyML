# Revision experiments (full)

- commit: 8af35e41c2f5c453dd15161c7c223552b0fbbd1e
- configs: config/jobs/2026-10-05_x1_calib_methods (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- quant_lit: 11 min

# (A) PTQ calibration estimators (mean / min / max over draws and calibration sets)

| model | method | runs_ok | runs_failed | f1_mean | f1_min | f1_max | f1_sd | far_mean | missed_mean | missed_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cic_near_iid | tflite_max | 15 | 0 | 77.67 | 29.16 | 84.99 | 13.62 | 15.51 | 349/85173 | 68-1643 |
| cic_near_iid | tflite_clip5 | 15 | 0 | 84.38 | 83.02 | 84.85 | 0.49 | 7.58 | 68/85173 | 53-87 |
| cic_near_iid | tflite_max_per_tensor | 15 | 0 | 76.66 | 29.2 | 83.69 | 13.32 | 16.16 | 336/85173 | 65-1459 |
| cic_near_iid | fq_max | 15 | 0 | 76.64 | 29.2 | 83.69 | 13.3 | 16.14 | 397/85173 | 65-1425 |
| cic_near_iid | fq_pct999 | 15 | 0 | 83.16 | 82.11 | 84.9 | 0.66 | 8.3 | 58/85173 | 51-64 |
| cic_near_iid | fq_pct9999 | 15 | 0 | 79.27 | 31.51 | 84.12 | 12.8 | 13.34 | 543/85173 | 51-7340 |
| cic_near_iid | fq_mse | 15 | 0 | 77.28 | 29.28 | 84.14 | 13.06 | 15.5 | 630/85173 | 67-1986 |
| cic_near_iid | fq_kl | 15 | 0 | 80.93 | 44.16 | 84.13 | 9.83 | 10.68 | 565/85173 | 52-4057 |
| cic_near_iid | fq_clip5_max | 15 | 0 | 82.93 | 82.09 | 83.54 | 0.4 | 8.37 | 239/85173 | 201-277 |
| ton_near_iid_v2 | tflite_max | 15 | 0 | 77.91 | 32.72 | 99.09 | 24.2 | 2.99 | 4779/16215 | 120-13024 |
| ton_near_iid_v2 | tflite_clip5 | 15 | 0 | 99.19 | 99.19 | 99.19 | 0.0 | 4.94 | 27/16215 | 27-27 |
| ton_near_iid_v2 | tflite_max_per_tensor | 15 | 0 | 81.66 | 30.25 | 99.33 | 23.51 | 4.44 | 3913/16215 | 38-13308 |
| ton_near_iid_v2 | fq_max | 15 | 0 | 81.0 | 36.06 | 99.31 | 21.78 | 4.48 | 4156/16215 | 47-12621 |
| ton_near_iid_v2 | fq_pct999 | 15 | 0 | 99.39 | 99.38 | 99.4 | 0.01 | 3.55 | 26/16215 | 24-29 |
| ton_near_iid_v2 | fq_pct9999 | 15 | 0 | 99.31 | 98.81 | 99.4 | 0.17 | 3.62 | 50/16215 | 16-207 |
| ton_near_iid_v2 | fq_mse | 15 | 0 | 88.68 | 34.62 | 99.39 | 19.5 | 4.72 | 2395/16215 | 20-12800 |
| ton_near_iid_v2 | fq_kl | 15 | 0 | 98.97 | 96.69 | 99.39 | 0.89 | 4.2 | 131/16215 | 24-802 |
| ton_near_iid_v2 | fq_clip5_max | 15 | 0 | 99.19 | 99.18 | 99.19 | 0.0 | 4.94 | 27/16215 | 27-30 |

