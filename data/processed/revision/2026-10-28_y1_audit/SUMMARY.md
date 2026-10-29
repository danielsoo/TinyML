# Revision experiments (full)

- commit: b14d9e205aaa7390693a8ac24288528ed2210fdf
- configs: config/jobs/2026-10-28_y1_audit (eval_split: )
- host: Linux 6.18.54.1-microsoft-standard-WSL2 x86_64, 16 cores
- audit: 3 min

# Pipeline audit

## cicids2017
- raw records 2,830,743; duplicates removed 331,200; conflicting-label vectors 719 (7,144 records)
- old pipeline: synthetic (SMOTE) share of training 0.375; rows with an invalid discrete value 0.0004; synthetic rows per client [255464, 255455, 255245, 255906]
- old threshold-validation slice: {'rows': 20000, 'part_of_client0_federated_training': True, 'synthetic_rows': 7554, 'synthetic_share': 0.3777, 'synthetic_share_of_attack_rows': 0.7456, 'synthetic_share_of_benign_rows': 0.0}
- negative values: {'Flow Duration': 115, 'Flow Bytes/s': 85, 'Flow Packets/s': 115, 'Flow IAT Mean': 115, 'Flow IAT Max': 115, 'Flow IAT Min': 2891, 'Fwd IAT Min': 17, 'Fwd Header Length': 35, 'Bwd Header Length': 22, 'Fwd Header Length.1': 35, 'Init_Win_bytes_forward': 1001189, 'Init_Win_bytes_backward': 1441552, 'min_seg_size_forward': 35}
- new test overlap with old test: {'new_test': 499621, 'old_test': 499909, 'new_test_records_that_were_in_old_test': 100169, 'share': 0.2005}
- max |z| 1148.7; top extreme: [{'id': 'Tuesday-WorkingHours.pcap_ISCX.csv:1267', 'attack': 'BENIGN', 'feature': 'Fwd Header Length', 'z': -1148.7, 'raw_value': -32212235446.19287}, {'id': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv:103957', 'attack': 'BENIGN', 'feature': 'Total Length of Fwd Packets', 'z': 1067.2, 'raw_value': 12899999.91769886}, {'id': 'Tuesday-WorkingHours.pcap_ISCX.csv:281248', 'attack': 'BENIGN', 'feature': 'Bwd Header Length', 'z': -609.3, 'raw_value': -1073741294.5827789}, {'id': 'Tuesday-WorkingHours.pcap_ISCX.csv:283', 'attack': 'BENIGN', 'feature': 'Bwd Header Length', 'z': -609.3, 'raw_value': -1073741294.5827789}, {'id': 'Tuesday-WorkingHours.pcap_ISCX.csv:292036', 'attack': 'BENIGN', 'feature': 'Bwd Header Length', 'z': -609.3, 'raw_value': -1073741294.5827789}]

## ton_iot
- raw records 211,043; duplicates removed 105,867; conflicting-label vectors 8 (314 records)
- old pipeline: synthetic (SMOTE) share of training 0.3513; rows with an invalid discrete value 0.0469; synthetic rows per client [11481, 11376, 11373, 11340]
- old threshold-validation slice: {'rows': 8000, 'part_of_client0_federated_training': True, 'synthetic_rows': 2807, 'synthetic_share': 0.3509, 'synthetic_share_of_attack_rows': 0.0, 'synthetic_share_of_benign_rows': 0.7101}
- negative values: {}
- new test overlap with old test: {'new_test': 21032, 'old_test': 21036, 'new_test_records_that_were_in_old_test': 4175, 'share': 0.1985}
- max |z| 272.4; top extreme: [{'id': 'train_test_network.csv:45094', 'attack': 'dos', 'feature': 'dst_ip_bytes', 'z': 272.4, 'raw_value': 86395521.84589386}, {'id': 'train_test_network.csv:95128', 'attack': 'normal', 'feature': 'http_response_body_len', 'z': 267.3, 'raw_value': 13424383.359280348}, {'id': 'train_test_network.csv:83129', 'attack': 'normal', 'feature': 'dst_pkts', 'z': 229.7, 'raw_value': 121941.99636473879}, {'id': 'train_test_network.csv:208219', 'attack': 'xss', 'feature': 'http_uri_special', 'z': 218.2, 'raw_value': 251.9999917439418}, {'id': 'train_test_network.csv:39547', 'attack': 'ddos', 'feature': 'missed_bytes', 'z': 186.9, 'raw_value': 909211720.8373642}]

