# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Models** | `models/tflite/saved_model_original.tflite`, `models/tflite/saved_model_qat_pruned_float32.tflite`, `models/tflite/saved_model_pruned_qat.tflite`, `models/tflite/saved_model_qat_ptq.tflite`, `models/tflite/saved_model_no_qat_ptq.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-10 18:06:54 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **Aggregation strategy** | FedAvgM (momentum=0.5) |
| **FL rounds** | 1 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | False |

## Summary

Models evaluated: 5 (Original, QAT, QAT→PTQ, Traditional PTQ where available)

Ratios per model: 11


---

## Original (TFLite)

**Model path**: `models/tflite/saved_model_original.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8988 | 0.0000 | 0.0000 | 0.0000 | 0.8988 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9066 | 0.5172 | 0.9862 | 0.6785 | 0.8977 | 0.9983 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9152 | 0.7064 | 0.9852 | 0.8229 | 0.8976 | 0.9959 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9241 | 0.8054 | 0.9852 | 0.8863 | 0.8980 | 0.9930 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9329 | 0.8655 | 0.9852 | 0.9215 | 0.8980 | 0.9891 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9422 | 0.9071 | 0.9852 | 0.9446 | 0.8991 | 0.9838 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9504 | 0.9356 | 0.9852 | 0.9598 | 0.8982 | 0.9759 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9590 | 0.9574 | 0.9852 | 0.9711 | 0.8978 | 0.9630 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9671 | 0.9740 | 0.9852 | 0.9796 | 0.8948 | 0.9380 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9768 | 0.9889 | 0.9852 | 0.9871 | 0.9008 | 0.8714 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9852 | 1.0000 | 0.9852 | 0.9926 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8988
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8988
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9066
- **Precision**: 0.5172
- **Recall**: 0.9862
- **F1-Score**: 0.6785
- **Normal Recall** (of actual normal, predicted normal): 0.8977
- **Normal Precision** (of predicted normal, actual normal): 0.9983

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9152
- **Precision**: 0.7064
- **Recall**: 0.9852
- **F1-Score**: 0.8229
- **Normal Recall** (of actual normal, predicted normal): 0.8976
- **Normal Precision** (of predicted normal, actual normal): 0.9959

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9241
- **Precision**: 0.8054
- **Recall**: 0.9852
- **F1-Score**: 0.8863
- **Normal Recall** (of actual normal, predicted normal): 0.8980
- **Normal Precision** (of predicted normal, actual normal): 0.9930

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9329
- **Precision**: 0.8655
- **Recall**: 0.9852
- **F1-Score**: 0.9215
- **Normal Recall** (of actual normal, predicted normal): 0.8980
- **Normal Precision** (of predicted normal, actual normal): 0.9891

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9422
- **Precision**: 0.9071
- **Recall**: 0.9852
- **F1-Score**: 0.9446
- **Normal Recall** (of actual normal, predicted normal): 0.8991
- **Normal Precision** (of predicted normal, actual normal): 0.9838

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9504
- **Precision**: 0.9356
- **Recall**: 0.9852
- **F1-Score**: 0.9598
- **Normal Recall** (of actual normal, predicted normal): 0.8982
- **Normal Precision** (of predicted normal, actual normal): 0.9759

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9590
- **Precision**: 0.9574
- **Recall**: 0.9852
- **F1-Score**: 0.9711
- **Normal Recall** (of actual normal, predicted normal): 0.8978
- **Normal Precision** (of predicted normal, actual normal): 0.9630

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9671
- **Precision**: 0.9740
- **Recall**: 0.9852
- **F1-Score**: 0.9796
- **Normal Recall** (of actual normal, predicted normal): 0.8948
- **Normal Precision** (of predicted normal, actual normal): 0.9380

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9768
- **Precision**: 0.9889
- **Recall**: 0.9852
- **F1-Score**: 0.9871
- **Normal Recall** (of actual normal, predicted normal): 0.9008
- **Normal Precision** (of predicted normal, actual normal): 0.8714

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9852
- **Precision**: 1.0000
- **Recall**: 0.9852
- **F1-Score**: 0.9926
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT+일반압축

**Model path**: `models/tflite/saved_model_qat_pruned_float32.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9002 | 0.0000 | 0.0000 | 0.0000 | 0.9002 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9072 | 0.5192 | 0.9724 | 0.6769 | 0.8999 | 0.9966 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9145 | 0.7086 | 0.9720 | 0.8197 | 0.9001 | 0.9923 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9209 | 0.8049 | 0.9720 | 0.8806 | 0.8990 | 0.9868 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9286 | 0.8660 | 0.9720 | 0.9159 | 0.8997 | 0.9797 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9366 | 0.9077 | 0.9720 | 0.9387 | 0.9011 | 0.9698 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9430 | 0.9356 | 0.9720 | 0.9534 | 0.8997 | 0.9554 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9504 | 0.9578 | 0.9720 | 0.9648 | 0.9001 | 0.9323 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9586 | 0.9761 | 0.9720 | 0.9740 | 0.9049 | 0.8898 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9648 | 0.9887 | 0.9720 | 0.9803 | 0.9004 | 0.7812 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9720 | 1.0000 | 0.9720 | 0.9858 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9002
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.9002
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9072
- **Precision**: 0.5192
- **Recall**: 0.9724
- **F1-Score**: 0.6769
- **Normal Recall** (of actual normal, predicted normal): 0.8999
- **Normal Precision** (of predicted normal, actual normal): 0.9966

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9145
- **Precision**: 0.7086
- **Recall**: 0.9720
- **F1-Score**: 0.8197
- **Normal Recall** (of actual normal, predicted normal): 0.9001
- **Normal Precision** (of predicted normal, actual normal): 0.9923

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9209
- **Precision**: 0.8049
- **Recall**: 0.9720
- **F1-Score**: 0.8806
- **Normal Recall** (of actual normal, predicted normal): 0.8990
- **Normal Precision** (of predicted normal, actual normal): 0.9868

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9286
- **Precision**: 0.8660
- **Recall**: 0.9720
- **F1-Score**: 0.9159
- **Normal Recall** (of actual normal, predicted normal): 0.8997
- **Normal Precision** (of predicted normal, actual normal): 0.9797

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9366
- **Precision**: 0.9077
- **Recall**: 0.9720
- **F1-Score**: 0.9387
- **Normal Recall** (of actual normal, predicted normal): 0.9011
- **Normal Precision** (of predicted normal, actual normal): 0.9698

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9430
- **Precision**: 0.9356
- **Recall**: 0.9720
- **F1-Score**: 0.9534
- **Normal Recall** (of actual normal, predicted normal): 0.8997
- **Normal Precision** (of predicted normal, actual normal): 0.9554

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9504
- **Precision**: 0.9578
- **Recall**: 0.9720
- **F1-Score**: 0.9648
- **Normal Recall** (of actual normal, predicted normal): 0.9001
- **Normal Precision** (of predicted normal, actual normal): 0.9323

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9586
- **Precision**: 0.9761
- **Recall**: 0.9720
- **F1-Score**: 0.9740
- **Normal Recall** (of actual normal, predicted normal): 0.9049
- **Normal Precision** (of predicted normal, actual normal): 0.8898

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9648
- **Precision**: 0.9887
- **Recall**: 0.9720
- **F1-Score**: 0.9803
- **Normal Recall** (of actual normal, predicted normal): 0.9004
- **Normal Precision** (of predicted normal, actual normal): 0.7812

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9720
- **Precision**: 1.0000
- **Recall**: 0.9720
- **F1-Score**: 0.9858
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT

**Model path**: `models/tflite/saved_model_pruned_qat.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9715 | 0.0000 | 0.0000 | 0.0000 | 0.9715 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9675 | 0.7850 | 0.9295 | 0.8512 | 0.9717 | 0.9920 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9630 | 0.8909 | 0.9288 | 0.9095 | 0.9716 | 0.9820 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9589 | 0.9337 | 0.9288 | 0.9313 | 0.9717 | 0.9696 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9549 | 0.9572 | 0.9288 | 0.9428 | 0.9723 | 0.9535 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9504 | 0.9707 | 0.9288 | 0.9493 | 0.9719 | 0.9318 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9458 | 0.9798 | 0.9288 | 0.9536 | 0.9712 | 0.9010 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9415 | 0.9868 | 0.9288 | 0.9569 | 0.9710 | 0.8540 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9375 | 0.9925 | 0.9288 | 0.9596 | 0.9721 | 0.7735 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9332 | 0.9967 | 0.9288 | 0.9616 | 0.9725 | 0.6029 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9288 | 1.0000 | 0.9288 | 0.9631 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9715
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.9715
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9675
- **Precision**: 0.7850
- **Recall**: 0.9295
- **F1-Score**: 0.8512
- **Normal Recall** (of actual normal, predicted normal): 0.9717
- **Normal Precision** (of predicted normal, actual normal): 0.9920

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9630
- **Precision**: 0.8909
- **Recall**: 0.9288
- **F1-Score**: 0.9095
- **Normal Recall** (of actual normal, predicted normal): 0.9716
- **Normal Precision** (of predicted normal, actual normal): 0.9820

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9589
- **Precision**: 0.9337
- **Recall**: 0.9288
- **F1-Score**: 0.9313
- **Normal Recall** (of actual normal, predicted normal): 0.9717
- **Normal Precision** (of predicted normal, actual normal): 0.9696

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9549
- **Precision**: 0.9572
- **Recall**: 0.9288
- **F1-Score**: 0.9428
- **Normal Recall** (of actual normal, predicted normal): 0.9723
- **Normal Precision** (of predicted normal, actual normal): 0.9535

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9504
- **Precision**: 0.9707
- **Recall**: 0.9288
- **F1-Score**: 0.9493
- **Normal Recall** (of actual normal, predicted normal): 0.9719
- **Normal Precision** (of predicted normal, actual normal): 0.9318

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9458
- **Precision**: 0.9798
- **Recall**: 0.9288
- **F1-Score**: 0.9536
- **Normal Recall** (of actual normal, predicted normal): 0.9712
- **Normal Precision** (of predicted normal, actual normal): 0.9010

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9415
- **Precision**: 0.9868
- **Recall**: 0.9288
- **F1-Score**: 0.9569
- **Normal Recall** (of actual normal, predicted normal): 0.9710
- **Normal Precision** (of predicted normal, actual normal): 0.8540

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9375
- **Precision**: 0.9925
- **Recall**: 0.9288
- **F1-Score**: 0.9596
- **Normal Recall** (of actual normal, predicted normal): 0.9721
- **Normal Precision** (of predicted normal, actual normal): 0.7735

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9332
- **Precision**: 0.9967
- **Recall**: 0.9288
- **F1-Score**: 0.9616
- **Normal Recall** (of actual normal, predicted normal): 0.9725
- **Normal Precision** (of predicted normal, actual normal): 0.6029

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9288
- **Precision**: 1.0000
- **Recall**: 0.9288
- **F1-Score**: 0.9631
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT+PTQ

**Model path**: `models/tflite/saved_model_qat_ptq.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9002 | 0.0000 | 0.0000 | 0.0000 | 0.9002 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9074 | 0.5196 | 0.9726 | 0.6774 | 0.9001 | 0.9966 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9148 | 0.7094 | 0.9720 | 0.8202 | 0.9005 | 0.9923 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9212 | 0.8056 | 0.9720 | 0.8810 | 0.8995 | 0.9868 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9287 | 0.8661 | 0.9720 | 0.9160 | 0.8999 | 0.9797 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9358 | 0.9064 | 0.9720 | 0.9380 | 0.8996 | 0.9698 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9424 | 0.9347 | 0.9720 | 0.9530 | 0.8981 | 0.9553 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9502 | 0.9575 | 0.9720 | 0.9647 | 0.8993 | 0.9322 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9575 | 0.9748 | 0.9720 | 0.9734 | 0.8996 | 0.8892 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9648 | 0.9887 | 0.9720 | 0.9803 | 0.8999 | 0.7811 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9720 | 1.0000 | 0.9720 | 0.9858 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9002
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.9002
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9074
- **Precision**: 0.5196
- **Recall**: 0.9726
- **F1-Score**: 0.6774
- **Normal Recall** (of actual normal, predicted normal): 0.9001
- **Normal Precision** (of predicted normal, actual normal): 0.9966

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9148
- **Precision**: 0.7094
- **Recall**: 0.9720
- **F1-Score**: 0.8202
- **Normal Recall** (of actual normal, predicted normal): 0.9005
- **Normal Precision** (of predicted normal, actual normal): 0.9923

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9212
- **Precision**: 0.8056
- **Recall**: 0.9720
- **F1-Score**: 0.8810
- **Normal Recall** (of actual normal, predicted normal): 0.8995
- **Normal Precision** (of predicted normal, actual normal): 0.9868

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9287
- **Precision**: 0.8661
- **Recall**: 0.9720
- **F1-Score**: 0.9160
- **Normal Recall** (of actual normal, predicted normal): 0.8999
- **Normal Precision** (of predicted normal, actual normal): 0.9797

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9358
- **Precision**: 0.9064
- **Recall**: 0.9720
- **F1-Score**: 0.9380
- **Normal Recall** (of actual normal, predicted normal): 0.8996
- **Normal Precision** (of predicted normal, actual normal): 0.9698

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9424
- **Precision**: 0.9347
- **Recall**: 0.9720
- **F1-Score**: 0.9530
- **Normal Recall** (of actual normal, predicted normal): 0.8981
- **Normal Precision** (of predicted normal, actual normal): 0.9553

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9502
- **Precision**: 0.9575
- **Recall**: 0.9720
- **F1-Score**: 0.9647
- **Normal Recall** (of actual normal, predicted normal): 0.8993
- **Normal Precision** (of predicted normal, actual normal): 0.9322

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9575
- **Precision**: 0.9748
- **Recall**: 0.9720
- **F1-Score**: 0.9734
- **Normal Recall** (of actual normal, predicted normal): 0.8996
- **Normal Precision** (of predicted normal, actual normal): 0.8892

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9648
- **Precision**: 0.9887
- **Recall**: 0.9720
- **F1-Score**: 0.9803
- **Normal Recall** (of actual normal, predicted normal): 0.8999
- **Normal Precision** (of predicted normal, actual normal): 0.7811

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9720
- **Precision**: 1.0000
- **Recall**: 0.9720
- **F1-Score**: 0.9858
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## noQAT+PTQ

**Model path**: `models/tflite/saved_model_no_qat_ptq.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8923 | 0.0000 | 0.0000 | 0.0000 | 0.8923 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9016 | 0.5040 | 0.9883 | 0.6676 | 0.8919 | 0.9985 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9111 | 0.6956 | 0.9875 | 0.8163 | 0.8920 | 0.9965 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9209 | 0.7972 | 0.9875 | 0.8822 | 0.8923 | 0.9940 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9299 | 0.8585 | 0.9875 | 0.9185 | 0.8915 | 0.9907 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9398 | 0.9015 | 0.9875 | 0.9425 | 0.8921 | 0.9862 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9494 | 0.9322 | 0.9875 | 0.9590 | 0.8922 | 0.9794 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9588 | 0.9551 | 0.9875 | 0.9710 | 0.8917 | 0.9683 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9683 | 0.9733 | 0.9875 | 0.9803 | 0.8916 | 0.9469 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9782 | 0.9883 | 0.9875 | 0.9879 | 0.8950 | 0.8882 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9875 | 1.0000 | 0.9875 | 0.9937 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8923
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8923
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9016
- **Precision**: 0.5040
- **Recall**: 0.9883
- **F1-Score**: 0.6676
- **Normal Recall** (of actual normal, predicted normal): 0.8919
- **Normal Precision** (of predicted normal, actual normal): 0.9985

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9111
- **Precision**: 0.6956
- **Recall**: 0.9875
- **F1-Score**: 0.8163
- **Normal Recall** (of actual normal, predicted normal): 0.8920
- **Normal Precision** (of predicted normal, actual normal): 0.9965

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9209
- **Precision**: 0.7972
- **Recall**: 0.9875
- **F1-Score**: 0.8822
- **Normal Recall** (of actual normal, predicted normal): 0.8923
- **Normal Precision** (of predicted normal, actual normal): 0.9940

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9299
- **Precision**: 0.8585
- **Recall**: 0.9875
- **F1-Score**: 0.9185
- **Normal Recall** (of actual normal, predicted normal): 0.8915
- **Normal Precision** (of predicted normal, actual normal): 0.9907

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9398
- **Precision**: 0.9015
- **Recall**: 0.9875
- **F1-Score**: 0.9425
- **Normal Recall** (of actual normal, predicted normal): 0.8921
- **Normal Precision** (of predicted normal, actual normal): 0.9862

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9494
- **Precision**: 0.9322
- **Recall**: 0.9875
- **F1-Score**: 0.9590
- **Normal Recall** (of actual normal, predicted normal): 0.8922
- **Normal Precision** (of predicted normal, actual normal): 0.9794

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9588
- **Precision**: 0.9551
- **Recall**: 0.9875
- **F1-Score**: 0.9710
- **Normal Recall** (of actual normal, predicted normal): 0.8917
- **Normal Precision** (of predicted normal, actual normal): 0.9683

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9683
- **Precision**: 0.9733
- **Recall**: 0.9875
- **F1-Score**: 0.9803
- **Normal Recall** (of actual normal, predicted normal): 0.8916
- **Normal Precision** (of predicted normal, actual normal): 0.9469

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9782
- **Precision**: 0.9883
- **Recall**: 0.9875
- **F1-Score**: 0.9879
- **Normal Recall** (of actual normal, predicted normal): 0.8950
- **Normal Precision** (of predicted normal, actual normal): 0.8882

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9875
- **Precision**: 1.0000
- **Recall**: 0.9875
- **F1-Score**: 0.9937
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

### Original (TFLite)

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.1076 | 0.0000 | 0.1076 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.60 | 0.9693 | 0.8316 | 0.9927 | 0.9737 | 0.7585 | 0.9204 |
| 80 | 20 | 0.55 | 0.9567 | 0.8932 | 0.9696 | 0.9761 | 0.9052 | 0.8815 |
| 70 | 30 | 0.55 | 0.9503 | 0.9161 | 0.9696 | 0.9598 | 0.9052 | 0.9273 |
| 60 | 40 | 0.55 | 0.9436 | 0.9278 | 0.9692 | 0.9388 | 0.9052 | 0.9515 |
| 50 | 50 | 0.50 | 0.9422 | 0.9446 | 0.8991 | 0.9838 | 0.9852 | 0.9071 |
| 40 | 60 | 0.50 | 0.9504 | 0.9598 | 0.8982 | 0.9759 | 0.9852 | 0.9356 |
| 30 | 70 | 0.50 | 0.9590 | 0.9711 | 0.8978 | 0.9630 | 0.9852 | 0.9574 |
| 20 | 80 | 0.50 | 0.9671 | 0.9796 | 0.8948 | 0.9380 | 0.9852 | 0.9740 |
| 10 | 90 | 0.50 | 0.9768 | 0.9871 | 0.9008 | 0.8714 | 0.9852 | 0.9889 |
| 0 | 100 | 0.15 | 0.9998 | 0.9999 | 0.0000 | 0.0000 | 0.9998 | 1.0000 |

### QAT+일반압축

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.0048 | 0.0000 | 0.0048 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.55 | 0.9566 | 0.7246 | 0.9995 | 0.9545 | 0.5708 | 0.9918 |
| 80 | 20 | 0.50 | 0.9145 | 0.8197 | 0.9001 | 0.9923 | 0.9720 | 0.7086 |
| 70 | 30 | 0.50 | 0.9209 | 0.8806 | 0.8990 | 0.9868 | 0.9720 | 0.8049 |
| 60 | 40 | 0.50 | 0.9286 | 0.9159 | 0.8997 | 0.9797 | 0.9720 | 0.8660 |
| 50 | 50 | 0.50 | 0.9366 | 0.9387 | 0.9011 | 0.9698 | 0.9720 | 0.9077 |
| 40 | 60 | 0.50 | 0.9430 | 0.9534 | 0.8997 | 0.9554 | 0.9720 | 0.9356 |
| 30 | 70 | 0.50 | 0.9504 | 0.9648 | 0.9001 | 0.9323 | 0.9720 | 0.9578 |
| 20 | 80 | 0.50 | 0.9586 | 0.9740 | 0.9049 | 0.8898 | 0.9720 | 0.9761 |
| 10 | 90 | 0.45 | 0.9732 | 0.9853 | 0.7644 | 0.9599 | 0.9964 | 0.9744 |
| 0 | 100 | 0.15 | 0.9997 | 0.9999 | 0.0000 | 0.0000 | 0.9997 | 1.0000 |

### QAT

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.9357 | 0.0000 | 0.9357 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.80 | 0.9742 | 0.8570 | 0.9963 | 0.9755 | 0.7745 | 0.9592 |
| 80 | 20 | 0.55 | 0.9635 | 0.9105 | 0.9722 | 0.9819 | 0.9285 | 0.8932 |
| 70 | 30 | 0.30 | 0.9591 | 0.9320 | 0.9696 | 0.9719 | 0.9346 | 0.9294 |
| 60 | 40 | 0.20 | 0.9561 | 0.9450 | 0.9647 | 0.9621 | 0.9430 | 0.9469 |
| 50 | 50 | 0.20 | 0.9536 | 0.9531 | 0.9642 | 0.9442 | 0.9430 | 0.9634 |
| 40 | 60 | 0.15 | 0.9531 | 0.9610 | 0.9355 | 0.9465 | 0.9647 | 0.9573 |
| 30 | 70 | 0.15 | 0.9560 | 0.9685 | 0.9356 | 0.9192 | 0.9647 | 0.9722 |
| 20 | 80 | 0.15 | 0.9592 | 0.9742 | 0.9368 | 0.8692 | 0.9647 | 0.9839 |
| 10 | 90 | 0.15 | 0.9621 | 0.9787 | 0.9387 | 0.7474 | 0.9647 | 0.9930 |
| 0 | 100 | 0.15 | 0.9648 | 0.9821 | 0.0000 | 0.0000 | 0.9648 | 1.0000 |

### QAT+PTQ

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.0051 | 0.0000 | 0.0051 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.55 | 0.9565 | 0.7240 | 0.9995 | 0.9544 | 0.5701 | 0.9916 |
| 80 | 20 | 0.50 | 0.9148 | 0.8202 | 0.9005 | 0.9923 | 0.9720 | 0.7094 |
| 70 | 30 | 0.50 | 0.9212 | 0.8810 | 0.8995 | 0.9868 | 0.9720 | 0.8056 |
| 60 | 40 | 0.50 | 0.9287 | 0.9160 | 0.8999 | 0.9797 | 0.9720 | 0.8661 |
| 50 | 50 | 0.50 | 0.9358 | 0.9380 | 0.8996 | 0.9698 | 0.9720 | 0.9064 |
| 40 | 60 | 0.50 | 0.9424 | 0.9530 | 0.8981 | 0.9553 | 0.9720 | 0.9347 |
| 30 | 70 | 0.50 | 0.9502 | 0.9647 | 0.8993 | 0.9322 | 0.9720 | 0.9575 |
| 20 | 80 | 0.50 | 0.9575 | 0.9734 | 0.8996 | 0.8892 | 0.9720 | 0.9748 |
| 10 | 90 | 0.45 | 0.9733 | 0.9853 | 0.7651 | 0.9601 | 0.9965 | 0.9745 |
| 0 | 100 | 0.15 | 0.9997 | 0.9999 | 0.0000 | 0.0000 | 0.9997 | 1.0000 |

### noQAT+PTQ

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.0001 | 0.0000 | 0.0001 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.50 | 0.9016 | 0.6676 | 0.8919 | 0.9985 | 0.9883 | 0.5040 |
| 80 | 20 | 0.50 | 0.9111 | 0.8163 | 0.8920 | 0.9965 | 0.9875 | 0.6956 |
| 70 | 30 | 0.50 | 0.9209 | 0.8822 | 0.8923 | 0.9940 | 0.9875 | 0.7972 |
| 60 | 40 | 0.50 | 0.9299 | 0.9185 | 0.8915 | 0.9907 | 0.9875 | 0.8585 |
| 50 | 50 | 0.50 | 0.9398 | 0.9425 | 0.8921 | 0.9862 | 0.9875 | 0.9015 |
| 40 | 60 | 0.50 | 0.9494 | 0.9590 | 0.8922 | 0.9794 | 0.9875 | 0.9322 |
| 30 | 70 | 0.50 | 0.9588 | 0.9710 | 0.8917 | 0.9683 | 0.9875 | 0.9551 |
| 20 | 80 | 0.50 | 0.9683 | 0.9803 | 0.8916 | 0.9469 | 0.9875 | 0.9733 |
| 10 | 90 | 0.50 | 0.9782 | 0.9879 | 0.8950 | 0.8882 | 0.9875 | 0.9883 |
| 0 | 100 | 0.15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

