# Baseline Ablation (Reviewer B)

| row_id | label | model_path | format | size_kb | latency_ms | accuracy | precision | recall | attack_recall | f1 | tp | tn | fp | fn | missed_attacks | false_alarms | false_alarm_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Centralized+cosine+focal | data/processed/revision/2026-10-03_toniot_full/baseline/models/a_centralized.h5 | keras | 2247.91 |  | 0.9931070545731128 | 0.9939144332431767 | 0.9971631205673759 | 0.9971631205673759 | 0.9955361265892927 | 16169 | 4722 | 99 | 46 | 46 | 99 | 0.020535158680771624 | 0.3 |
| b | FL+cosine+focal | data/processed/revision/2026-10-03_toniot_full/baseline/models/b_fl.h5 | keras | 765.73 |  | 0.9821734169994295 | 0.9896142433234422 | 0.9872340425531915 | 0.9872340425531915 | 0.9884227100120404 | 16008 | 4653 | 168 | 207 | 207 | 168 | 0.03484754200373367 | 0.3 |
| c | FL+compression+PTQ | data/processed/revision/2026-10-03_toniot_full/baseline/tflite/saved_model_qat_ptq.tflite | tflite | 64.81 | 0.0019 | 0.9896368130823351 | 0.9901942759085616 | 0.9964230650632131 | 0.9964230650632131 | 0.9932989056928563 | 16157 | 4661 | 160 | 58 | 58 | 160 | 0.03318813524165111 | 0.3 |
| d | FL+compression+QAT | data/processed/revision/2026-10-03_toniot_full/baseline/tflite/saved_model_pruned_qat.tflite | tflite | 55.18 | 0.0019 | 0.9890663624263167 | 0.9899466683013547 | 0.9959296947271046 | 0.9959296947271046 | 0.992929168716183 | 16149 | 4657 | 164 | 66 | 66 | 164 | 0.034017838622692385 | 0.3 |
