# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirichlet | 4 | [{'client': 0, 'total': 9373, 'attack_pct': 0.11, 'normal_pct': 99.89}, {'client': 1, 'total': 1981913, 'attack_pct': 59.53, 'normal_pct': 40.47}, {'client': 2, 'total': 6730, 'attack_pct': 2.59, 'normal_pct': 97.41}, {'client': 3, 'total': 727504, 'attack_pct': 25.13, 'normal_pct': 74.87}] | 0.942733577511107 | 0.7490683886427105 | 0.9983093233771265 | 0.9983093233771265 | 0.8559133507141923 | 144 | 28484 | 144 | 0.06867983488291347 |
