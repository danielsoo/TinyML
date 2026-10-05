# Revision experiments (full)

- commit: 0bfcb0c6d53db418f5fe533384fe73daadd75743
- configs: config/jobs/2026-10-05_q_attack_counts (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- attack_counts: 1 min

# Flows per attack type (train after balancing/SMOTE, test)

| entry | type | train | test |
| --- | --- | --- | --- |
| cic_near_iid | BENIGN | 1362760 | 414736 |
| cic_near_iid | DoS Hulk | 138362 | 34487 |
| cic_near_iid | DDoS | 102447 | 25569 |
| cic_near_iid | PortScan | 72537 | 18282 |
| cic_near_iid | DoS GoldenEye | 8276 | 2006 |
| cic_near_iid | FTP-Patator | 4697 | 1236 |
| cic_near_iid | DoS Slowhttptest | 4157 | 1071 |
| cic_near_iid | DoS slowloris | 4311 | 1063 |
| cic_near_iid | SSH-Patator | 2569 | 650 |
| cic_near_iid | Bot | 1585 | 368 |
| cic_near_iid | Web Attack � Brute Force | 1169 | 301 |
| cic_near_iid | Web Attack � XSS | 532 | 120 |
| cic_near_iid | Infiltration | 25 | 11 |
| cic_near_iid | Heartbleed | 6 | 5 |
| cic_near_iid | Web Attack � Sql Injection | 17 | 4 |
| cic_near_iid | SMOTE | 1022070 | 0 |

