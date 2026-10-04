# Compression pipeline combinations (mean / min / max over fine-tuning draws)

| model | pipeline | description | draws_ok | draws_failed | size_kb | gzip_kb | weight_sparsity | f1_mean | f1_min | f1_max | recall_mean | far_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton_near_iid | fp32 | no compression (reference) | 1 | 0 | 720.44 | 672.3 | 0.0 | 99.4 | 99.4 | 99.4 | 99.62 | 2.76 |
| ton_near_iid | q_ptq | INT8 PTQ | 3 | 0 | 206.38 | 155.5 | 0.039 | 98.87 | 98.37 | 99.26 | 98.55 | 2.69 |
| ton_near_iid | q_ptq_clip | INT8 PTQ, clipped calibration | 3 | 0 | 206.39 | 155.9 | 0.039 | 99.34 | 99.34 | 99.34 | 99.54 | 2.91 |
| ton_near_iid | q_qat | QAT fine-tune, no pruning | 3 | 0 | 186.24 | 125.0 | 0.09 | 99.1 | 98.98 | 99.33 | 99.02 | 2.77 |
| ton_near_iid | s_ptq | structured 50% -> FT -> PTQ | 3 | 0 | 64.84 | 47.0 | 0.034 | 98.5 | 97.44 | 99.21 | 97.66 | 2.09 |
| ton_near_iid | s_ptq_clip | structured 50% -> FT -> PTQ (clipped) | 3 | 0 | 64.84 | 47.2 | 0.034 | 99.37 | 99.34 | 99.39 | 99.42 | 2.29 |
| ton_near_iid | s_qat | structured 50% -> FT -> QAT (deployed) | 3 | 0 | 55.19 | 38.9 | 0.065 | 99.08 | 98.88 | 99.38 | 99.03 | 2.9 |
| ton_near_iid | s_qat_clipft | structured 50% -> FT -> QAT on clipped inputs | 3 | 0 | 55.19 | 39.1 | 0.065 | 99.4 | 99.34 | 99.44 | 99.55 | 2.55 |
| ton_near_iid | s_kd_qat | structured 50% -> KD fine-tune -> QAT | 3 | 0 | 55.19 | 38.6 | 0.066 | 99.13 | 98.97 | 99.43 | 99.15 | 3.01 |
| ton_near_iid | s_kd_ptq_clip | structured 50% -> KD fine-tune -> PTQ (clipped) | 3 | 0 | 64.84 | 47.0 | 0.034 | 99.37 | 99.36 | 99.38 | 99.52 | 2.63 |
| ton_near_iid | o_qat_s_qat | QAT first -> structured 50% -> FT -> QAT | 3 | 0 | 55.19 | 39.0 | 0.064 | 99.12 | 98.98 | 99.36 | 99.03 | 2.67 |
| ton_near_iid | u50_ptq_clip | magnitude 50% -> PTQ (clipped) | 3 | 0 | 206.41 | 124.4 | 0.5 | 99.4 | 99.39 | 99.41 | 99.44 | 2.18 |
| ton_near_iid | u50_pqat | magnitude 50% -> sparsity-preserving QAT | 3 | 0 | 186.25 | 100.0 | 0.5 | 99.12 | 99.0 | 99.35 | 99.03 | 2.68 |
| ton_near_iid | u80_ptq_clip | magnitude 80% -> PTQ (clipped) | 3 | 0 | 206.41 | 71.2 | 0.8 | 99.34 | 99.31 | 99.37 | 99.42 | 2.48 |
| ton_near_iid | u80_pqat | magnitude 80% -> sparsity-preserving QAT | 3 | 0 | 186.25 | 56.3 | 0.8 | 98.85 | 98.51 | 99.36 | 98.74 | 3.52 |
| ton_near_iid | u80_qat | magnitude 80% -> standard QAT | 3 | 0 | 186.25 | 76.3 | 0.619 | 99.07 | 98.89 | 99.33 | 98.98 | 2.86 |
| ton_near_iid | su_pqat | structured 50% -> FT -> magnitude 50% -> PQAT | 3 | 0 | 54.98 | 31.2 | 0.5 | 98.95 | 98.7 | 99.37 | 98.81 | 3.01 |
| cic_near_iid | fp32 | no compression (reference) | 1 | 0 | 802.55 | 754.5 | 0.0 | 85.7 | 85.7 | 85.7 | 99.94 | 6.84 |
| cic_near_iid | q_ptq | INT8 PTQ | 3 | 0 | 226.91 | 149.8 | 0.171 | 82.36 | 81.04 | 83.21 | 99.81 | 8.75 |
| cic_near_iid | q_ptq_clip | INT8 PTQ, clipped calibration | 3 | 0 | 226.91 | 150.1 | 0.171 | 83.97 | 83.27 | 84.35 | 99.93 | 7.82 |
| cic_near_iid | q_qat | QAT fine-tune, no pruning | 3 | 0 | 206.76 | 98.8 | 0.398 | 90.51 | 89.27 | 91.99 | 99.32 | 4.15 |
| cic_near_iid | s_ptq | structured 50% -> FT -> PTQ | 3 | 0 | 75.12 | 47.0 | 0.156 | 84.39 | 79.37 | 87.27 | 99.41 | 7.52 |
| cic_near_iid | s_ptq_clip | structured 50% -> FT -> PTQ (clipped) | 3 | 0 | 75.12 | 47.1 | 0.156 | 87.67 | 86.86 | 88.09 | 99.03 | 5.52 |
| cic_near_iid | s_qat | structured 50% -> FT -> QAT (deployed) | 3 | 0 | 65.45 | 34.4 | 0.33 | 88.37 | 87.64 | 89.28 | 99.41 | 5.26 |
| cic_near_iid | s_qat_clipft | structured 50% -> FT -> QAT on clipped inputs | 3 | 0 | 65.45 | 34.5 | 0.334 | 90.36 | 88.54 | 92.44 | 98.63 | 4.07 |
| cic_near_iid | s_kd_qat | structured 50% -> KD fine-tune -> QAT | 3 | 0 | 65.45 | 34.0 | 0.349 | 86.04 | 85.61 | 86.56 | 99.68 | 6.58 |
| cic_near_iid | s_kd_ptq_clip | structured 50% -> KD fine-tune -> PTQ (clipped) | 3 | 0 | 75.12 | 46.9 | 0.17 | 84.85 | 84.23 | 85.16 | 99.72 | 7.26 |
| cic_near_iid | o_qat_s_qat | QAT first -> structured 50% -> FT -> QAT | 3 | 0 | 65.45 | 34.4 | 0.326 | 88.64 | 86.65 | 90.58 | 99.19 | 5.07 |
| cic_near_iid | u50_ptq_clip | magnitude 50% -> PTQ (clipped) | 3 | 0 | 226.94 | 128.0 | 0.501 | 89.72 | 89.48 | 89.95 | 99.42 | 4.56 |
| cic_near_iid | u50_pqat | magnitude 50% -> sparsity-preserving QAT | 3 | 0 | 206.76 | 89.8 | 0.532 | 90.7 | 89.95 | 91.71 | 99.43 | 4.07 |
| cic_near_iid | u80_ptq_clip | magnitude 80% -> PTQ (clipped) | 3 | 0 | 226.94 | 75.9 | 0.8 | 87.98 | 87.57 | 88.18 | 98.92 | 5.33 |
| cic_near_iid | u80_pqat | magnitude 80% -> sparsity-preserving QAT | 3 | 0 | 206.76 | 56.5 | 0.8 | 89.77 | 88.99 | 91.1 | 99.28 | 4.5 |
| cic_near_iid | u80_qat | magnitude 80% -> standard QAT | 3 | 0 | 206.76 | 61.7 | 0.755 | 89.77 | 89.31 | 90.52 | 99.33 | 4.52 |
| cic_near_iid | su_pqat | structured 50% -> FT -> magnitude 50% -> PQAT | 3 | 0 | 65.33 | 31.2 | 0.511 | 88.66 | 88.05 | 89.86 | 99.18 | 5.05 |
