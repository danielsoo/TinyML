# Revision experiments (full)

- commit: 8fdf60b298803147ecaca0c255726dfe42e3b6e4
- configs: config/paper_v12_toniot_v2 (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- baseline_ablation: 17 min
- fixed_lr: 5 min
- non_iid: 7 min
- qat_fl: 5 min

# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-05_u_toniot_v2/baseline/models/a_centralized.h5 | keras | 2283.91 |  | 0.9951036318691766 | 0.994293778377715 | 0.9993832870798643 | 0.9993832870798643 | 0.9968320364162028 | 16205 | 4728 | 93 | 10 | 10 | 93 | 0.019290603609209707 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-05_u_toniot_v2/baseline/models/b_fl.h5 | keras | 780.76 |  | 0.9903974139570261 | 0.9890660313969825 | 0.9985815602836879 | 0.9985815602836879 | 0.9938010188424476 | 16192 | 4642 | 179 | 23 | 23 | 179 | 0.03712922630159718 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-05_u_toniot_v2/baseline/tflite/saved_model_no_qat_ptq.tflite | tflite | 66.34 | 0.0029 | 0.9823635672181023 | 0.9909519087754091 | 0.9861239592969473 | 0.9861239592969473 | 0.988532039195079 | 15990 | 4675 | 146 | 225 | 225 | 146 | 0.030284173408006636 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-05_u_toniot_v2/baseline/tflite/saved_model_traditional_qat.tflite | tflite | 56.69 | 0.0019 | 0.9876877733409394 | 0.9902900688298918 | 0.9937711995066296 | 0.9937711995066296 | 0.9920275802628743 | 16114 | 4663 | 158 | 101 | 101 | 158 | 0.03277328355113047 | 0.3 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 49871, 'attack_pct': 0.43, 'normal_pct': 99.57}, {'client': 1, 'total': 62856, 'attack_pct': 92.75, 'normal_pct': 7.25}, {'client': 2, 'total': 8594, 'attack_pct': 65.46, 'normal_pct': 34.54}, {'client': 3, 'total': 8389, 'attack_pct': 8.5, 'normal_pct': 91.5}] | 0.9781327248526336 | 0.9724704612247346 | 0.9999383287079864 | 0.9999383287079864 | 0.9860131354901484 | 1 | 459 | 1 | 0.09520846297448662 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_balanced | 4 | [{'client': 0, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 1, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 2, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 3, 'total': 32426, 'attack_pct': 50.0, 'normal_pct': 50.0}] | 0.9897318881916715 | 0.9882209337808971 | 0.9985815602836879 | 0.9985815602836879 | 0.9933742331288343 | 23 | 193 | 23 | 0.04003318813524165 |

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_balanced | 4 | [{'client': 0, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 1, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 2, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 3, 'total': 32426, 'attack_pct': 50.0, 'normal_pct': 50.0}] | 0.9813652785700704 | 0.9861136712749616 | 0.9897625655257478 | 0.9897625655257478 | 0.9879347491535857 | 166 | 226 | 166 | 0.046878241028832195 |

