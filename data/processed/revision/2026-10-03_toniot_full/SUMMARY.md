# Revision experiments (full)

- commit: 61ed363331980897989a5ddd6b1553f26da9a6fd
- configs: config/paper_v12_toniot (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 17 min
- non_iid: 7 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-03_toniot_full/baseline/models/a_centralized.h5 | keras | 2247.91 |  | 0.9931070545731128 | 0.9939144332431767 | 0.9971631205673759 | 0.9971631205673759 | 0.9955361265892927 | 16169 | 4722 | 99 | 46 | 46 | 99 | 0.020535158680771624 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-03_toniot_full/baseline/models/b_fl.h5 | keras | 765.73 |  | 0.9821734169994295 | 0.9896142433234422 | 0.9872340425531915 | 0.9872340425531915 | 0.9884227100120404 | 16008 | 4653 | 168 | 207 | 207 | 168 | 0.03484754200373367 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-03_toniot_full/baseline/tflite/saved_model_qat_ptq.tflite | tflite | 64.81 | 0.0019 | 0.9896368130823351 | 0.9901942759085616 | 0.9964230650632131 | 0.9964230650632131 | 0.9932989056928563 | 16157 | 4661 | 160 | 58 | 58 | 160 | 0.03318813524165111 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-03_toniot_full/baseline/tflite/saved_model_pruned_qat.tflite | tflite | 55.18 | 0.0019 | 0.9890663624263167 | 0.9899466683013547 | 0.9959296947271046 | 0.9959296947271046 | 0.992929168716183 | 16149 | 4657 | 164 | 66 | 66 | 164 | 0.034017838622692385 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 49871, 'attack_pct': 0.43, 'normal_pct': 99.57}, {'client': 1, 'total': 62856, 'attack_pct': 92.75, 'normal_pct': 7.25}, {'client': 2, 'total': 8594, 'attack_pct': 65.46, 'normal_pct': 34.54}, {'client': 3, 'total': 8389, 'attack_pct': 8.5, 'normal_pct': 91.5}] | 0.9741395702605058 | 0.9762932344538326 | 0.9905026210299106 | 0.9905026210299106 | 0.983346598910182 | 154 | 390 | 154 | 0.08089607965152458 |

