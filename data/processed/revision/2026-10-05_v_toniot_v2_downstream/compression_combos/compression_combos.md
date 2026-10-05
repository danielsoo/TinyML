# Compression pipeline combinations (mean / min / max over fine-tuning draws)

| model | pipeline | description | draws_ok | draws_failed | size_kb | gzip_kb | weight_sparsity | f1_mean | f1_min | f1_max | recall_mean | far_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton2_near_iid | fp32 | no compression (reference) | 1 | 0 | 732.44 | 683.8 | 0.0 | 99.41 | 99.41 | 99.41 | 99.88 | 3.61 |
| ton2_near_iid | q_ptq | INT8 PTQ | 3 | 0 | 209.38 | 159.7 | 0.037 | 99.1 | 98.92 | 99.4 | 99.28 | 3.62 |
| ton2_near_iid | q_ptq_clip | INT8 PTQ, clipped calibration | 3 | 0 | 209.39 | 160.0 | 0.037 | 99.18 | 99.18 | 99.19 | 99.83 | 4.94 |
| ton2_near_iid | q_qat | QAT fine-tune, no pruning | 3 | 0 | 189.24 | 130.6 | 0.075 | 99.22 | 99.07 | 99.52 | 99.05 | 2.05 |
| ton2_near_iid | s_ptq | structured 50% -> FT -> PTQ | 3 | 0 | 66.34 | 48.6 | 0.034 | 98.32 | 96.89 | 99.19 | 97.21 | 1.76 |
| ton2_near_iid | s_ptq_clip | structured 50% -> FT -> PTQ (clipped) | 3 | 0 | 66.34 | 48.7 | 0.034 | 99.42 | 99.4 | 99.44 | 99.38 | 1.8 |
| ton2_near_iid | s_qat | structured 50% -> FT -> QAT (deployed) | 3 | 0 | 56.69 | 40.4 | 0.058 | 99.12 | 98.89 | 99.51 | 98.93 | 2.31 |
| ton2_near_iid | s_qat_clipft | structured 50% -> FT -> QAT on clipped inputs | 3 | 0 | 56.69 | 40.6 | 0.057 | 99.49 | 99.47 | 99.51 | 99.48 | 1.72 |
| ton2_near_iid | s_kd_qat | structured 50% -> KD fine-tune -> QAT | 3 | 0 | 56.69 | 40.2 | 0.06 | 99.19 | 99.01 | 99.49 | 99.14 | 2.59 |
| ton2_near_iid | s_kd_ptq_clip | structured 50% -> KD fine-tune -> PTQ (clipped) | 3 | 0 | 66.34 | 48.5 | 0.035 | 99.42 | 99.34 | 99.47 | 99.66 | 2.79 |
| ton2_near_iid | o_qat_s_qat | QAT first -> structured 50% -> FT -> QAT | 3 | 0 | 56.69 | 40.5 | 0.057 | 99.17 | 98.97 | 99.48 | 98.98 | 2.16 |
| ton2_near_iid | u50_ptq_clip | magnitude 50% -> PTQ (clipped) | 3 | 0 | 209.41 | 126.7 | 0.5 | 99.47 | 99.46 | 99.47 | 99.38 | 1.51 |
| ton2_near_iid | u50_pqat | magnitude 50% -> sparsity-preserving QAT | 3 | 0 | 189.25 | 104.0 | 0.5 | 99.18 | 99.03 | 99.42 | 98.98 | 2.08 |
| ton2_near_iid | u80_ptq_clip | magnitude 80% -> PTQ (clipped) | 3 | 0 | 209.41 | 72.4 | 0.8 | 99.38 | 99.36 | 99.39 | 99.31 | 1.87 |
| ton2_near_iid | u80_pqat | magnitude 80% -> sparsity-preserving QAT | 3 | 0 | 189.25 | 58.0 | 0.8 | 99.03 | 98.82 | 99.43 | 98.82 | 2.58 |
| ton2_near_iid | u80_qat | magnitude 80% -> standard QAT | 3 | 0 | 189.25 | 80.2 | 0.597 | 99.06 | 98.86 | 99.46 | 98.77 | 2.15 |
| ton2_near_iid | su_pqat | structured 50% -> FT -> magnitude 50% -> PQAT | 3 | 0 | 56.48 | 32.5 | 0.5 | 99.01 | 98.74 | 99.46 | 98.69 | 2.26 |
