# Revision experiments (full)

- commit: 0193cb2864d45d5d6c3a930a316fa496978ecb9d
- configs: config/paper_v12_float (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 321 min
- non_iid: 164 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/a_centralized.h5 | keras | 2493.91 |  | 0.9506930261307558 | 0.7759973004031154 | 0.9989668087304663 | 0.9989668087304663 | 0.8734774328992552 | 85085 | 390175 | 24561 | 88 | 88 | 24561 | 0.05922080552447822 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-03_b_v3_float_cic/baseline/models/b_fl.h5 | keras | 850.76 |  | 0.9431676565134854 | 0.7500969333803313 | 0.9993894778861846 | 0.9993894778861846 | 0.8569817722360095 | 85121 | 386377 | 28359 | 52 | 52 | 28359 | 0.06837843833185447 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-03_b_v3_float_cic/baseline/tflite/saved_model_no_qat_ptq.tflite | tflite | 75.09 | 0.0022 | 0.22026408806402764 | 0.17931210180354307 | 0.9999060735209515 | 0.9999060735209515 | 0.3040917506208413 | 85165 | 24947 | 389789 | 8 | 8 | 389789 | 0.9398484819258516 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-03_b_v3_float_cic/baseline/tflite/saved_model_traditional_qat.tflite | tflite | 65.44 | 0.0026 | 0.9690023584292341 | 0.874531009793698 | 0.9550914022049241 | 0.9550914022049241 | 0.913037622339948 | 81348 | 403065 | 11671 | 3825 | 3825 | 11671 | 0.02814079317927549 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 9373, 'attack_pct': 0.11, 'normal_pct': 99.89}, {'client': 1, 'total': 1981913, 'attack_pct': 59.53, 'normal_pct': 40.47}, {'client': 2, 'total': 6730, 'attack_pct': 2.59, 'normal_pct': 97.41}, {'client': 3, 'total': 727504, 'attack_pct': 25.13, 'normal_pct': 74.87}] | 0.942733577511107 | 0.7490683886427105 | 0.9983093233771265 | 0.9983093233771265 | 0.8559133507141923 | 144 | 28484 | 144 | 0.06867983488291347 |

