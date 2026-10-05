# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_balanced | 4 | [{'client': 0, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 1, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 2, 'total': 32428, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 3, 'total': 32426, 'attack_pct': 50.0, 'normal_pct': 50.0}] | 0.9897318881916715 | 0.9882209337808971 | 0.9985815602836879 | 0.9985815602836879 | 0.9933742331288343 | 23 | 193 | 23 | 0.04003318813524165 |
