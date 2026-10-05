# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-05_u_toniot_v2/baseline/models/a_centralized.h5 | keras | 2283.91 |  | 0.9951036318691766 | 0.994293778377715 | 0.9993832870798643 | 0.9993832870798643 | 0.9968320364162028 | 16205 | 4728 | 93 | 10 | 10 | 93 | 0.019290603609209707 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-05_u_toniot_v2/baseline/models/b_fl.h5 | keras | 780.76 |  | 0.9903974139570261 | 0.9890660313969825 | 0.9985815602836879 | 0.9985815602836879 | 0.9938010188424476 | 16192 | 4642 | 179 | 23 | 23 | 179 | 0.03712922630159718 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-05_u_toniot_v2/baseline/tflite/saved_model_no_qat_ptq.tflite | tflite | 66.34 | 0.0029 | 0.9823635672181023 | 0.9909519087754091 | 0.9861239592969473 | 0.9861239592969473 | 0.988532039195079 | 15990 | 4675 | 146 | 225 | 225 | 146 | 0.030284173408006636 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-05_u_toniot_v2/baseline/tflite/saved_model_traditional_qat.tflite | tflite | 56.69 | 0.0019 | 0.9876877733409394 | 0.9902900688298918 | 0.9937711995066296 | 0.9937711995066296 | 0.9920275802628743 | 16114 | 4663 | 158 | 101 | 101 | 158 | 0.03277328355113047 | 0.3 |
