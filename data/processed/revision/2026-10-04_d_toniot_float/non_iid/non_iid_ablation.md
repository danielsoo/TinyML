# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 49871, 'attack_pct': 0.43, 'normal_pct': 99.57}, {'client': 1, 'total': 62856, 'attack_pct': 92.75, 'normal_pct': 7.25}, {'client': 2, 'total': 8594, 'attack_pct': 65.46, 'normal_pct': 34.54}, {'client': 3, 'total': 8389, 'attack_pct': 8.5, 'normal_pct': 91.5}] | 0.9762787602205742 | 0.9892292367077574 | 0.9798951588035769 | 0.9798951588035769 | 0.984540074975989 | 326 | 173 | 326 | 0.03588467123003526 |
