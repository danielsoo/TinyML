# Revision experiments (full)

- commit: 1d791eac746147fee3f946e787684e16ba8bca2e
- configs: config/jobs/2026-10-04_f_robustness (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- robustness: 5 min

# FGSM / PGD robustness (transfer from float FL model)

| dataset | attack | display_name | original_accuracy | adversarial_accuracy | attack_success_rate |
| --- | --- | --- | --- | --- | --- |
| cic | fgsm | b_fl | 0.9426 | 0.6858 | 0.2568 |
| cic | fgsm | a_centralized | 0.95065 | 0.3075 | 0.64315 |
| cic | fgsm | fp32 | 0.9426 | 0.6858 | 0.2568 |
| cic | fgsm | ptq_only | 0.93865 | 0.7038 | 0.23485 |
| cic | fgsm | prune_ft_client_ptq | 0.9484 | 0.3248 | 0.6236 |
| cic | fgsm | prune_ft_client_qat | 0.9419 | 0.4429 | 0.499 |
| cic | fgsm | prune_ft_pooled_qat | 0.9633 | 0.4474 | 0.5159 |
| cic | pgd | b_fl | 0.9426 | 0.5555 | 0.3871 |
| cic | pgd | a_centralized | 0.95065 | 0.29 | 0.66065 |
| cic | pgd | fp32 | 0.9426 | 0.5555 | 0.3871 |
| cic | pgd | ptq_only | 0.93865 | 0.6044 | 0.33425 |
| cic | pgd | prune_ft_client_ptq | 0.9484 | 0.23915 | 0.70925 |
| cic | pgd | prune_ft_client_qat | 0.9419 | 0.4556 | 0.4863 |
| cic | pgd | prune_ft_pooled_qat | 0.9633 | 0.46185 | 0.50145 |
| ton_iot | fgsm | b_fl | 0.9909 | 0.22975 | 0.76115 |
| ton_iot | fgsm | a_centralized | 0.992 | 0.2296 | 0.7624 |
| ton_iot | fgsm | fp32 | 0.9909 | 0.22975 | 0.76115 |
| ton_iot | fgsm | ptq_only | 0.9754 | 0.2297 | 0.7457 |
| ton_iot | fgsm | prune_ft_client_ptq | 0.8609 | 0.2295 | 0.6314 |
| ton_iot | fgsm | prune_ft_client_qat | 0.98075 | 0.22975 | 0.751 |
| ton_iot | fgsm | prune_ft_pooled_qat | 0.9873 | 0.56155 | 0.42575 |
| ton_iot | pgd | b_fl | 0.9909 | 0.2295 | 0.7614 |
| ton_iot | pgd | a_centralized | 0.992 | 0.22965 | 0.76235 |
| ton_iot | pgd | fp32 | 0.9909 | 0.2295 | 0.7614 |
| ton_iot | pgd | ptq_only | 0.9754 | 0.2295 | 0.7459 |
| ton_iot | pgd | prune_ft_client_ptq | 0.8609 | 0.2295 | 0.6314 |
| ton_iot | pgd | prune_ft_client_qat | 0.98075 | 0.22955 | 0.7512 |
| ton_iot | pgd | prune_ft_pooled_qat | 0.9873 | 0.55565 | 0.43165 |

