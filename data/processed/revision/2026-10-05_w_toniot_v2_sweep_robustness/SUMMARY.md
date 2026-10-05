# Revision experiments (full)

- commit: 1e63a0f157196835da76102a696b030183451fa3
- configs: config/jobs/2026-10-05_w_toniot_v2_sweep_robustness (eval_split: )
- host: Linux 6.18.40.1-microsoft-standard-WSL2 x86_64, 16 cores
- compression_ablation: 0 min
- robustness: 0 min

# Compression fine-tuning ablation

| model | variant | description | ft_data | size_kb | accuracy | precision | attack_recall | f1 | false_alarm_rate | fp | fn | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton2_near_iid | fp32 | FL model, no compression | - | 732.44 | 0.9908252519490397 | 0.9893708002443494 | 0.9988282454517422 | 0.9940770293079638 | 0.03609209707529558 | 174 | 19 | 0.3 |
| ton2_near_iid | ptq_only | INT8 PTQ only (no pruning, no fine-tune) | - | 209.38 | 0.9838847689674843 | 0.9885524372230429 | 0.9905642923219241 | 0.9895573422049717 | 0.038581207218419414 | 186 | 153 | 0.3 |
| ton2_near_iid | prune_noft_ptq_r30 | prune 30% -> PTQ (no fine-tune) | - | 113.64 | 0.9807472903593839 | 0.9771821803694314 | 0.9983348751156337 | 0.987645282328178 | 0.07840696950840075 | 378 | 27 | 0.3 |
| ton2_near_iid | prune_ft_client_ptq_r30 | prune 30% -> fine-tune -> PTQ | client | 113.67 | 0.8639475185396464 | 0.9565752581549614 | 0.8626580326857848 | 0.9071924249302807 | 0.13171541174030285 | 635 | 2227 | 0.3 |
| ton2_near_iid | prune_ft_client_qat_r30 | prune 30% -> fine-tune -> QAT | client | 99.83 | 0.9807472903593839 | 0.9940007499062617 | 0.9809435707678076 | 0.9874289971133253 | 0.019912881144990666 | 96 | 309 | 0.3 |
| ton2_near_iid | prune_noft_ptq_r50 | prune 50% -> PTQ (no fine-tune) | - | 66.34 | 0.9246529758509222 | 0.9169992019154031 | 0.9920444033302498 | 0.9530467754836035 | 0.3020120306990251 | 1456 | 129 | 0.3 |
| ton2_near_iid | prune_ft_client_ptq_r50 | prune 50% -> fine-tune -> PTQ | client | 66.34 | 0.8570545731127591 | 0.955951394642364 | 0.8539007092198582 | 0.902048926675136 | 0.1323376892760838 | 638 | 2369 | 0.3 |
| ton2_near_iid | prune_ft_client_qat_r50 | prune 50% -> fine-tune -> QAT | client | 56.69 | 0.9796539266020156 | 0.9931896282411746 | 0.9803268578476719 | 0.9867163252638114 | 0.022609417133374818 | 109 | 319 | 0.3 |
| ton2_near_iid | prune_noft_ptq_r70 | prune 70% -> PTQ (no fine-tune) | - | 31.23 | 0.9140521011599163 | 0.899905623716205 | 0.9996916435399321 | 0.947177749211172 | 0.37398879900435594 | 1803 | 5 | 0.3 |
| ton2_near_iid | prune_ft_client_ptq_r70 | prune 70% -> fine-tune -> PTQ | client | 31.23 | 0.8479273626164671 | 0.9555508889822204 | 0.8418748072772124 | 0.8951181928461361 | 0.13171541174030285 | 635 | 2564 | 0.3 |
| ton2_near_iid | prune_ft_client_qat_r70 | prune 70% -> fine-tune -> QAT | client | 25.81 | 0.9793687012740064 | 0.992386895475819 | 0.9807585568917668 | 0.9865384615384616 | 0.02530595312175897 | 122 | 312 | 0.3 |
| ton2_near_iid | prune_noft_ptq_r85 | prune 85% -> PTQ (no fine-tune) | - | 13.66 | 0.7708214489446663 | 0.7708214489446663 | 1.0 | 0.8705806555528711 | 1.0 | 4821 | 0 | 0.3 |
| ton2_near_iid | prune_ft_client_ptq_r85 | prune 85% -> fine-tune -> PTQ | client | 13.66 | 0.8241585852823731 | 0.9554585152838428 | 0.8096207215541166 | 0.8765147721582373 | 0.1269446172993155 | 612 | 3087 | 0.3 |
| ton2_near_iid | prune_ft_client_qat_r85 | prune 85% -> fine-tune -> QAT | client | 11.38 | 0.9792260886100019 | 0.9917102966841187 | 0.9812519272278755 | 0.9864533928516073 | 0.027587637419622484 | 133 | 304 | 0.3 |
| ton2_near_iid | prune_noft_ptq_r90 | prune 90% -> PTQ (no fine-tune) | - | 9.41 | 0.7708214489446663 | 0.7708214489446663 | 1.0 | 0.8705806555528711 | 1.0 | 4821 | 0 | 0.3 |
| ton2_near_iid | prune_ft_client_ptq_r90 | prune 90% -> fine-tune -> PTQ | client | 9.41 | 0.874738543449325 | 0.9599024654565158 | 0.8740055504162813 | 0.9149423803221538 | 0.12279610039410911 | 592 | 2043 | 0.3 |
| ton2_near_iid | prune_ft_client_qat_r90 | prune 90% -> fine-tune -> QAT | client | 8.16 | 0.9797490017113519 | 0.9916547300242885 | 0.9819919827320382 | 0.9867997025285077 | 0.027795063264882805 | 134 | 292 | 0.3 |

# FGSM / PGD robustness (transfer from float FL model)

| dataset | attack | display_name | original_accuracy | adversarial_accuracy | attack_success_rate |
| --- | --- | --- | --- | --- | --- |
| ton_iot_v2 | fgsm | best_text_alpha05_test | 0.99105 | 0.2295 | 0.76155 |
| ton_iot_v2 | fgsm | a_centralized | 0.99525 | 0.2301 | 0.76515 |
| ton_iot_v2 | fgsm | fp32 | 0.99105 | 0.2295 | 0.76155 |
| ton_iot_v2 | fgsm | ptq_only | 0.98405 | 0.2295 | 0.75455 |
| ton_iot_v2 | fgsm | prune_ft_client_ptq | 0.8572 | 0.2295 | 0.6277 |
| ton_iot_v2 | fgsm | prune_ft_client_qat | 0.97985 | 0.2299 | 0.74995 |
| ton_iot_v2 | fgsm | prune_ft_pooled_qat | 0.9871 | 0.91145 | 0.07565 |
| ton_iot_v2 | pgd | best_text_alpha05_test | 0.99105 | 0.2293 | 0.76175 |
| ton_iot_v2 | pgd | a_centralized | 0.99525 | 0.22965 | 0.7656 |
| ton_iot_v2 | pgd | fp32 | 0.99105 | 0.2293 | 0.76175 |
| ton_iot_v2 | pgd | ptq_only | 0.98405 | 0.2294 | 0.75465 |
| ton_iot_v2 | pgd | prune_ft_client_ptq | 0.8572 | 0.2295 | 0.6277 |
| ton_iot_v2 | pgd | prune_ft_client_qat | 0.97985 | 0.2297 | 0.75015 |
| ton_iot_v2 | pgd | prune_ft_pooled_qat | 0.9871 | 0.9112 | 0.0759 |

