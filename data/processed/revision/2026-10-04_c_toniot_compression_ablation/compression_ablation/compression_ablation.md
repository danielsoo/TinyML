# Compression fine-tuning ablation

| model | variant | description | ft_data | size_kb | accuracy | precision | attack_recall | f1 | false_alarm_rate | fp | fn | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toniot_near_iid | fp32 | FL model, no compression | - | 720.44 | 0.8479749001711352 | 0.8355415785946281 | 0.9995066296638915 | 0.9101988093900932 | 0.661688446380419 | 3190 | 8 | 0.3 |
| toniot_near_iid | ptq_only | INT8 PTQ only (no pruning, no fine-tune) | - | 206.38 | 0.8515402167712492 | 0.8389251320285803 | 0.9992599444958372 | 0.9120999746685806 | 0.6453018046048538 | 3111 | 12 | 0.3 |
| toniot_near_iid | prune_noft_ptq | prune 50% -> PTQ (no fine-tune) | - | 64.83 | 0.8138904734740445 | 0.8055141579731744 | 1.0 | 0.8922822946760215 | 0.8120721841941506 | 3915 | 0 | 0.3 |
| toniot_near_iid | ftonly_pooled_ptq | fine-tune 3 ep (no pruning) -> PTQ | pooled | 206.41 | 0.9902072637383533 | 0.9909230297454769 | 0.9964230650632131 | 0.9936654366543666 | 0.030699025098527278 | 148 | 58 | 0.3 |
| toniot_near_iid | prune_ft_pooled_ptq | prune 50% -> fine-tune -> PTQ | pooled | 64.84 | 0.9903498764023578 | 0.9915275049115914 | 0.9959913660191181 | 0.9937544226686768 | 0.02862476664592408 | 138 | 65 | 0.3 |
| toniot_near_iid | prune_ft_pooled_qat | prune -> fine-tune -> QAT 2 ep (deploy recipe) | pooled | 55.18 | 0.9889712873169804 | 0.9898253141281029 | 0.9959296947271046 | 0.9928681217337842 | 0.03443269031321303 | 166 | 66 | 0.3 |
| toniot_near_iid | ftonly_client_ptq | fine-tune 3 ep (no pruning) -> PTQ | client | 206.41 | 0.9831241680927933 | 0.992852703542573 | 0.9851988899167438 | 0.989010989010989 | 0.023853972204936735 | 115 | 240 | 0.3 |
| toniot_near_iid | prune_ft_client_ptq | prune 50% -> fine-tune -> PTQ | client | 64.84 | 0.9798916143753565 | 0.992945436384068 | 0.9808818994757941 | 0.9868768032761456 | 0.023439120514416097 | 113 | 310 | 0.3 |
| toniot_near_iid | prune_ft_client_qat | prune -> fine-tune -> QAT 2 ep (deploy recipe) | client | 55.19 | 0.9841224567408252 | 0.9923730390029144 | 0.9869873573851372 | 0.9896728711891659 | 0.02551337896701929 | 123 | 211 | 0.3 |
| toniot_dirichlet | fp32 | FL model, no compression | - | 720.53 | 0.8443145084616848 | 0.8320332546443601 | 0.9998766574159729 | 0.9082658749054648 | 0.6789047915370255 | 3273 | 2 | 0.3 |
| toniot_dirichlet | ptq_only | INT8 PTQ only (no pruning, no fine-tune) | - | 206.41 | 0.843886670469671 | 0.8316491408053347 | 0.9998766574159729 | 0.9080369644357322 | 0.6807716241443684 | 3282 | 2 | 0.3 |
| toniot_dirichlet | prune_noft_ptq | prune 50% -> PTQ (no fine-tune) | - | 64.84 | 0.8066647651644799 | 0.7996842313005723 | 0.999568300955905 | 0.8885234218677192 | 0.8421489317568969 | 4060 | 7 | 0.3 |
| toniot_dirichlet | ftonly_pooled_ptq | fine-tune 3 ep (no pruning) -> PTQ | pooled | 206.41 | 0.9843601445141662 | 0.990429735737219 | 0.9892691951896392 | 0.9898491252969671 | 0.03215100601534951 | 155 | 174 | 0.3 |
| toniot_dirichlet | prune_ft_pooled_ptq | prune 50% -> fine-tune -> PTQ | pooled | 64.85 | 0.9813652785700704 | 0.9902708062217265 | 0.9855072463768116 | 0.9878832838773492 | 0.03256585770587015 | 157 | 235 | 0.3 |
| toniot_dirichlet | prune_ft_pooled_qat | prune -> fine-tune -> QAT 2 ep (deploy recipe) | pooled | 55.19 | 0.9867370222475755 | 0.9897357098955132 | 0.9930928152944805 | 0.9914114206556872 | 0.034640116158473344 | 167 | 112 | 0.3 |
| toniot_dirichlet | ftonly_client_ptq | fine-tune 3 ep (no pruning) -> PTQ | client | 206.41 | 0.9822684921087659 | 0.9782634947470112 | 0.9991982732038236 | 0.9886200689507886 | 0.07467330429371499 | 360 | 13 | 0.3 |
| toniot_dirichlet | prune_ft_client_ptq | prune 50% -> fine-tune -> PTQ | client | 64.85 | 0.9873074729035939 | 0.9862195121951219 | 0.9974714770274438 | 0.9918135827073432 | 0.046878241028832195 | 226 | 41 | 0.3 |
| toniot_dirichlet | prune_ft_client_qat | prune -> fine-tune -> QAT 2 ep (deploy recipe) | client | 55.19 | 0.9857387335995437 | 0.9829459246222007 | 0.9988282454517422 | 0.9908234430441699 | 0.05828666251814976 | 281 | 19 | 0.3 |
