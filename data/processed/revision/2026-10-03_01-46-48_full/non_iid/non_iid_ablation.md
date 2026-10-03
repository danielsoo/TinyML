# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 9373, 'attack_pct': 0.11, 'normal_pct': 99.89}, {'client': 1, 'total': 1981913, 'attack_pct': 59.53, 'normal_pct': 40.47}, {'client': 2, 'total': 6730, 'attack_pct': 2.59, 'normal_pct': 97.41}, {'client': 3, 'total': 727504, 'attack_pct': 25.13, 'normal_pct': 74.87}] | 0.8491065373898049 | 0.5400184068236725 | 0.7715708029539877 | 0.7715708029539877 | 0.6353550832177196 | 19456 | 55977 | 19456 | 0.13497019790903128 |
