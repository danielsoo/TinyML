# Revision experiments (full)

- commit: 91978616fbc5c1f750cd208fb95890850e3a960f
- configs: config/paper_v12_float (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- fixed_lr: 68 min

# Non-IID FL Ablation

| strategy | num_clients | client_distribution | accuracy | precision | recall | attack_recall | f1 | fn | fp | missed_attacks | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_balanced | 4 | [{'client': 0, 'total': 681380, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 1, 'total': 681380, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 2, 'total': 681380, 'attack_pct': 50.0, 'normal_pct': 50.0}, {'client': 3, 'total': 681380, 'attack_pct': 50.0, 'normal_pct': 50.0}] | 0.9500569103576851 | 0.773992427276368 | 0.9984032498561751 | 0.9984032498561751 | 0.8719910172732912 | 136 | 24831 | 136 | 0.059871822074765636 |

