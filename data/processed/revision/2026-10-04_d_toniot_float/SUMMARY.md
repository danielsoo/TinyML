# Revision experiments (full)

- commit: ec67ca93eb1f126a1a7c9b1eb45258a3c67a2507
- configs: config/paper_v12_toniot (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 16 min
- non_iid: 6 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-04_d_toniot_float/baseline/models/a_centralized.h5 | keras | 2247.91 |  | 0.9919661532610763 | 0.9932374277634329 | 0.9963613937711995 | 0.9963613937711995 | 0.9947969582217296 | 16156 | 4711 | 110 | 59 | 59 | 110 | 0.02281684297863514 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-04_d_toniot_float/baseline/models/b_fl.h5 | keras | 768.76 |  | 0.9907777143943716 | 0.9918339780192792 | 0.9962380511871723 | 0.9962380511871723 | 0.9940311365454434 | 16154 | 4688 | 133 | 61 | 61 | 133 | 0.027587637419622484 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-04_d_toniot_float/baseline/tflite/saved_model_no_qat_ptq.tflite | tflite | 64.84 | 0.0019 | 0.8057615516257843 | 0.7987879982263388 | 0.9998766574159729 | 0.9998766574159729 | 0.8880915863277827 | 16213 | 737 | 4084 | 2 | 2 | 4084 | 0.8471271520431446 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-04_d_toniot_float/baseline/tflite/saved_model_traditional_qat.tflite | tflite | 55.19 | 0.0018 | 0.8931355771059136 | 0.8800544217687075 | 0.997286463151403 | 0.997286463151403 | 0.9350101185313674 | 16171 | 2617 | 2204 | 44 | 44 | 2204 | 0.457166562953744 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 49871, 'attack_pct': 0.43, 'normal_pct': 99.57}, {'client': 1, 'total': 62856, 'attack_pct': 92.75, 'normal_pct': 7.25}, {'client': 2, 'total': 8594, 'attack_pct': 65.46, 'normal_pct': 34.54}, {'client': 3, 'total': 8389, 'attack_pct': 8.5, 'normal_pct': 91.5}] | 0.9762787602205742 | 0.9892292367077574 | 0.9798951588035769 | 0.9798951588035769 | 0.984540074975989 | 326 | 173 | 326 | 0.03588467123003526 |

