# Revision experiments (full)

- commit: d3a26be094bf0ea2334fe49727bf6751ca8d6980
- configs: config/paper_v12 (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 330 min
- non_iid: 163 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-03_v2_cic_full/baseline/models/a_centralized.h5 | keras | 2493.91 |  | 0.957356238835468 | 0.8000507485409795 | 0.9995068859849953 | 0.9995068859849953 | 0.8887253366739742 | 85131 | 393460 | 21276 | 42 | 42 | 21276 | 0.05130010416264805 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-03_v2_cic_full/baseline/models/b_fl.h5 | keras | 847.73 |  | 0.3474032273873845 | 0.20698616283705598 | 0.9996712573233302 | 0.9996712573233302 | 0.34296083411039435 | 85145 | 88525 | 326211 | 28 | 28 | 326211 | 0.7865509625400254 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-03_v2_cic_full/baseline/tflite/saved_model_qat_ptq.tflite | tflite | 75.06 | 0.0021 | 0.9196433750942672 | 0.6901482245170449 | 0.9588484613668651 | 0.9588484613668651 | 0.8026062985548408 | 81668 | 378070 | 36666 | 3505 | 3505 | 36666 | 0.08840804752903052 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-03_v2_cic_full/baseline/tflite/saved_model_pruned_qat.tflite | tflite | 65.43 | 0.0025 | 0.9484846242016047 | 0.8103714846851364 | 0.9107581040940204 | 0.9107581040940204 | 0.857637218969911 | 77572 | 396584 | 18152 | 7601 | 7601 | 18152 | 0.043767601558581844 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 9373, 'attack_pct': 0.11, 'normal_pct': 99.89}, {'client': 1, 'total': 1981913, 'attack_pct': 59.53, 'normal_pct': 40.47}, {'client': 2, 'total': 6730, 'attack_pct': 2.59, 'normal_pct': 97.41}, {'client': 3, 'total': 727504, 'attack_pct': 25.13, 'normal_pct': 74.87}] | 0.18061887263481954 | 0.17214004353370832 | 1.0 | 1.0 | 0.2937192436745856 | 0 | 409616 | 0 | 0.9876547972686239 |

