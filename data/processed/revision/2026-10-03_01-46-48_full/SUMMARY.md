# Revision experiments (full)

- commit: 
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 322 min
- non_iid: 155 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-03_01-46-48_full/baseline/models/a_centralized.h5 | keras | 2493.91 |  | 0.9500429078092213 | 0.7736293306424487 | 0.9991429208786822 | 0.9991429208786822 | 0.8720423826944163 | 85100 | 389835 | 24901 | 73 | 73 | 24901 | 0.06004060414335867 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-03_01-46-48_full/baseline/models/b_fl.h5 | keras | 847.73 |  | 0.9054527924082183 | 0.6477249699936091 | 0.9757434867857184 | 0.9757434867857184 | 0.7785964895844556 | 83107 | 369537 | 45199 | 2066 | 2066 | 45199 | 0.1089825816905212 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-03_01-46-48_full/baseline/tflite/saved_model_qat_ptq.tflite | tflite | 75.06 | 0.002 | 0.9634493477812962 | 0.8705646456701637 | 0.9226515445035398 | 0.9226515445035398 | 0.8958516204785627 | 78585 | 403052 | 11684 | 6588 | 6588 | 11684 | 0.028172138420585625 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-03_01-46-48_full/baseline/tflite/saved_model_pruned_qat.tflite | tflite | 65.43 | 0.0024 | 0.9661218341738196 | 0.8704546194855536 | 0.9412372465452666 | 0.9412372465452666 | 0.9044631977977345 | 80168 | 402805 | 11931 | 5005 | 5005 | 11931 | 0.028767698005478185 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 9373, 'attack_pct': 0.11, 'normal_pct': 99.89}, {'client': 1, 'total': 1981913, 'attack_pct': 59.53, 'normal_pct': 40.47}, {'client': 2, 'total': 6730, 'attack_pct': 2.59, 'normal_pct': 97.41}, {'client': 3, 'total': 727504, 'attack_pct': 25.13, 'normal_pct': 74.87}] | 0.8491065373898049 | 0.5400184068236725 | 0.7715708029539877 | 0.7715708029539877 | 0.6353550832177196 | 19456 | 55977 | 19456 | 0.13497019790903128 |

