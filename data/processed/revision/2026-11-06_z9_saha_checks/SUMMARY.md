# Revision experiments (full)

- commit: 8501f6577dd32f29b89350da2176c6fb816f41e3
- configs: config/jobs/2026-11-06_z9_saha_checks (eval_split: )
- host: Linux 6.18.54.1-microsoft-standard-WSL2 x86_64, 16 cores
- saha_checks: 9 min

# Checks for Dr. Saha's completion list

See saha_checks.json for every number.

- provenance ton_random: new test {'records': 21032, 'in_old_test': 0, 'in_old_training_pool': 21032, 'dropped_by_old_dedup': 0}; client train {'records': 75716, 'in_old_test': 18927, 'in_old_training_pool': 56789, 'dropped_by_old_dedup': 0}; client val {'records': 8412, 'in_old_test': 2105, 'in_old_training_pool': 6307, 'dropped_by_old_dedup': 0}
- provenance ton_hosts: new test {'records': 22866, 'in_old_test': 0, 'in_old_training_pool': 22866, 'dropped_by_old_dedup': 0}; client train {'records': 68937, 'in_old_test': 13797, 'in_old_training_pool': 55140, 'dropped_by_old_dedup': 0}; client val {'records': 7660, 'in_old_test': 1538, 'in_old_training_pool': 6122, 'dropped_by_old_dedup': 0}
- provenance cic_random: new test {'records': 499038, 'in_old_test': 0, 'in_old_training_pool': 499038, 'dropped_by_old_dedup': 0}; client train {'records': 1529825, 'in_old_test': 382551, 'in_old_training_pool': 1147274, 'dropped_by_old_dedup': 0}; client val {'records': 199616, 'in_old_test': 49739, 'in_old_training_pool': 149877, 'dropped_by_old_dedup': 0}
- ton F8_s1: Table-2 export recall/FAR@1% 0.9828/0.0112; per-tensor (clip5_pct9999) 0.9828/0.0112 (193792 B)
- ton P80_s1: Table-2 export recall/FAR@1% 0.9642/0.0091; per-tensor (tfl_clip5_pt) 0.9798/0.0100 (57280 B)
- ton S80_s1: Table-2 export recall/FAR@1% 0.9817/0.0085; per-tensor (tfl_clip5_pt) 0.9767/0.0093 (57280 B)
- ton F8_s2: Table-2 export recall/FAR@1% 0.9844/0.0124; per-tensor (tfl_clip5_pt) 0.9840/0.0127 (193024 B)
- ton P80_s2: Table-2 export recall/FAR@1% 0.9547/0.0095; per-tensor (kl) 0.9442/0.0095 (58056 B)
- ton S80_s2: Table-2 export recall/FAR@1% 0.9803/0.0093; per-tensor (clip5_pct9999) 0.9802/0.0093 (58056 B)
- ton F8_s3: Table-2 export recall/FAR@1% 0.9838/0.0118; per-tensor (pct999) 0.9838/0.0118 (193800 B)
- ton P80_s3: Table-2 export recall/FAR@1% 0.9732/0.0100; per-tensor (clip5_pct9999) 0.9732/0.0100 (58056 B)
- ton S80_s3: Table-2 export recall/FAR@1% 0.9700/0.0116; per-tensor (pct9999) 0.9700/0.0116 (58056 B)
- cic F8_s1: Table-2 export recall/FAR@1% 0.9933/0.0098; per-tensor (tfl_clip5_pt) 0.9971/0.0092 (210952 B)
- cic P80_s1: Table-2 export recall/FAR@1% 0.9887/0.0098; per-tensor (pct999) 0.9887/0.0098 (67016 B)
- cic S80_s1: Table-2 export recall/FAR@1% 0.9929/0.0097; per-tensor (pct9999) 0.9929/0.0097 (67016 B)
- cic F8_s2: Table-2 export recall/FAR@1% 0.9961/0.0093; per-tensor (tfl_clip5_pt) 0.9957/0.0094 (210952 B)
- cic P80_s2: Table-2 export recall/FAR@1% 0.9904/0.0085; per-tensor (clip5_pct9999) 0.9895/0.0081 (67016 B)
- cic S80_s2: Table-2 export recall/FAR@1% 0.9963/0.0090; per-tensor (tfl_clip5_pt) 0.9970/0.0096 (66248 B)
- cic F8_s3: Table-2 export recall/FAR@1% 0.9945/0.0090; per-tensor (pct9999) 0.9872/0.0083 (211720 B)
- cic P80_s3: Table-2 export recall/FAR@1% 0.9935/0.0098; per-tensor (tfl_clip5_pt) 0.9955/0.0095 (66248 B)
- cic S80_s3: Table-2 export recall/FAR@1% 0.9970/0.0088; per-tensor (tfl_clip5_pt) 0.9971/0.0099 (66248 B)
- ton_hosts F8_s1: Table-2 export recall/FAR@1% 0.9818/0.0000; per-tensor (kl) 0.9818/0.0000 (193800 B)
- ton_hosts P80_s1: Table-2 export recall/FAR@1% 0.9399/0.0000; per-tensor (kl) 0.9399/0.0000 (58056 B)
- ton_hosts S80_s1: Table-2 export recall/FAR@1% 0.9672/0.0000; per-tensor (tfl_clip5_pt) 0.9695/0.0000 (57288 B)
- ton_hosts F8_s2: Table-2 export recall/FAR@1% 0.9787/0.0000; per-tensor (tfl_clip5_pt) 0.9796/0.0000 (193032 B)
- ton_hosts P80_s2: Table-2 export recall/FAR@1% 0.9642/0.0000; per-tensor (clip5_pct9999) 0.9350/0.0000 (58056 B)
- ton_hosts S80_s2: Table-2 export recall/FAR@1% 0.9717/0.0000; per-tensor (tfl_clip5_pt) 0.9710/0.0000 (57288 B)
- ton_hosts F8_s3: Table-2 export recall/FAR@1% 0.9818/0.0000; per-tensor (tfl_clip5_pt) 0.9829/0.0000 (193032 B)
- ton_hosts P80_s3: Table-2 export recall/FAR@1% 0.9394/0.0000; per-tensor (pct999) 0.9394/0.0000 (58056 B)
- ton_hosts S80_s3: Table-2 export recall/FAR@1% 0.9540/0.0000; per-tensor (mse) 0.9540/0.0000 (58056 B)

