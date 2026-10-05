# Revision experiments (full)

- commit: 8435f2d5f74e13f342cffe26cb5ee0c196d33fbf
- configs: config/jobs/2026-10-05_s_ton_sweep (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- ton_sweep: 39 min

# TON_IoT federated training sweep (selection on validation)

| variant | split | n | attacks | roc_auc | pr_auc | f1_fixed | recall_fixed | far_fixed | fn_fixed | threshold_target | recall_target | far_at_target | fn_target | missed_by_type_fixed | overrides | model_path | config_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | val | 8414 | 6485 | 0.998479 | 0.999317 | 99.362 | 99.722 | 3.37 | 18 | 0.205113 | 99.907 | 6.843 | 6 | {"password": 4, "dos": 3, "backdoor": 3, "xss": 3, "ddos": 2, "mitm": 1, "injection": 1, "ransomware": 1} | {} |  |  |
| alpha05 | val | 8414 | 6485 | 0.998791 | 0.999414 | 99.348 | 99.815 | 3.784 | 12 | 0.291354 | 99.907 | 3.784 | 6 | {"dos": 4, "xss": 3, "ddos": 2, "password": 2, "ransomware": 1} | {"federated": {"focal_loss_alpha": 0.5}} |  |  |
| alpha065 | val | 8414 | 6485 | 0.998355 | 0.999267 | 99.059 | 99.877 | 5.962 | 8 | 0.268777 | 99.907 | 7.724 | 6 | {"xss": 3, "ddos": 2, "password": 2, "dos": 1} | {"federated": {"focal_loss_alpha": 0.65}} |  |  |
| nosmote | val | 8414 | 6485 | 0.998735 | 0.999568 | 99.279 | 99.815 | 4.251 | 12 | 0.277089 | 99.907 | 4.303 | 6 | {"password": 4, "dos": 3, "ddos": 2, "xss": 2, "ransomware": 1} | {"data": {"use_smote": false}} |  |  |
| text | val | 8414 | 6485 | 0.999033 | 0.999416 | 99.455 | 99.907 | 3.37 | 6 | 0.307241 | 99.907 | 3.214 | 6 | {"xss": 3, "dos": 2, "ransomware": 1} | {"data": {"text_features": true}} |  |  |
| rounds100 | val | 8414 | 6485 | 0.998069 | 0.998059 | 99.355 | 99.784 | 3.629 | 14 | 0.258192 | 99.907 | 5.184 | 6 | {"dos": 6, "xss": 3, "ddos": 2, "password": 2, "backdoor": 1} | {"federated": {"num_rounds": 100}} |  |  |
| text_alpha05 | val | 8414 | 6485 | 0.998531 | 0.998869 | 99.394 | 99.923 | 3.836 | 5 | 0.369168 | 99.907 | 3.007 | 6 | {"xss": 3, "dos": 2} | {"data": {"text_features": true}, "federated": {"focal_loss_alpha": 0.5}} |  |  |
| text_nosmote | val | 8414 | 6485 | 0.998875 | 0.9996 | 99.41 | 99.954 | 3.836 | 3 | 0.355105 | 99.907 | 3.059 | 6 | {"xss": 2, "dos": 1} | {"data": {"text_features": true, "use_smote": false}} |  |  |
| best_text_alpha05 | test | 21036 | 16215 | 0.99895 | 0.999591 | 99.408 | 99.883 | 3.609 | 19 | 0.272824 | 99.907 | 4.522 | 15 | {"mitm": 5, "scanning": 4, "ransomware": 3, "dos": 3, "xss": 2, "injection": 1, "ddos": 1} |  | data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/models/best_text_alpha05_test.h5 | data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/configs/best_text_alpha05_test.yaml |
| reference | test | 21036 | 16215 | 0.998596 | 0.999367 | 99.403 | 99.624 | 2.759 | 61 | 0.173703 | 99.901 | 7.156 | 16 | {"dos": 16, "mitm": 14, "ddos": 10, "xss": 7, "ransomware": 4, "backdoor": 4, "injection": 4, "password": 2} |  |  |  |

