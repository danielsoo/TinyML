# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Models** | `models/tflite/saved_model_original.tflite`, `models/tflite/saved_model_qat_pruned_float32.tflite`, `models/tflite/saved_model_pruned_qat.tflite`, `models/tflite/saved_model_qat_ptq.tflite`, `models/tflite/saved_model_no_qat_ptq.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-10 19:48:09 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **Aggregation strategy** | FedAvgM (momentum=0.5) |
| **FL rounds** | 80 |
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

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | F1-Score | Attack Recall | Attack Precision | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9916 | 0.0000 | 0.0000 | 0.0000 | 0.9916 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9327 | 0.5440 | 0.4012 | 0.8445 | 0.9918 | 0.9371 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8731 | 0.5565 | 0.3982 | 0.9237 | 0.9918 | 0.8683 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.8136 | 0.5617 | 0.3982 | 0.9532 | 0.9916 | 0.7936 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.7540 | 0.5642 | 0.3982 | 0.9677 | 0.9911 | 0.7118 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.6952 | 0.5664 | 0.3982 | 0.9806 | 0.9921 | 0.6224 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.6353 | 0.5671 | 0.3982 | 0.9850 | 0.9909 | 0.5233 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.5762 | 0.5681 | 0.3982 | 0.9911 | 0.9916 | 0.4139 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.5169 | 0.5688 | 0.3982 | 0.9950 | 0.9920 | 0.2918 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.4577 | 0.5693 | 0.3982 | 0.9981 | 0.9934 | 0.1550 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.3982 | 0.5696 | 0.3982 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9916
- **F1-Score**: 0.0000 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.0000
- **Attack Precision** (of predicted attack, actual attack): 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.9916
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9327
- **F1-Score**: 0.5440 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4012
- **Attack Precision** (of predicted attack, actual attack): 0.8445
- **Normal Recall** (of actual normal, predicted normal): 0.9918
- **Normal Precision** (of predicted normal, actual normal): 0.9371

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8731
- **F1-Score**: 0.5565 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9237
- **Normal Recall** (of actual normal, predicted normal): 0.9918
- **Normal Precision** (of predicted normal, actual normal): 0.8683

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.8136
- **F1-Score**: 0.5617 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9532
- **Normal Recall** (of actual normal, predicted normal): 0.9916
- **Normal Precision** (of predicted normal, actual normal): 0.7936

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.7540
- **F1-Score**: 0.5642 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9677
- **Normal Recall** (of actual normal, predicted normal): 0.9911
- **Normal Precision** (of predicted normal, actual normal): 0.7118

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.6952
- **F1-Score**: 0.5664 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9806
- **Normal Recall** (of actual normal, predicted normal): 0.9921
- **Normal Precision** (of predicted normal, actual normal): 0.6224

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.6353
- **F1-Score**: 0.5671 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9850
- **Normal Recall** (of actual normal, predicted normal): 0.9909
- **Normal Precision** (of predicted normal, actual normal): 0.5233

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.5762
- **F1-Score**: 0.5681 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9911
- **Normal Recall** (of actual normal, predicted normal): 0.9916
- **Normal Precision** (of predicted normal, actual normal): 0.4139

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.5169
- **F1-Score**: 0.5688 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9950
- **Normal Recall** (of actual normal, predicted normal): 0.9920
- **Normal Precision** (of predicted normal, actual normal): 0.2918

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.4577
- **F1-Score**: 0.5693 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 0.9981
- **Normal Recall** (of actual normal, predicted normal): 0.9934
- **Normal Precision** (of predicted normal, actual normal): 0.1550

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.3982
- **F1-Score**: 0.5696 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.3982
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT+Prune only

**Model path**: `models/tflite/saved_model_qat_pruned_float32.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | F1-Score | Attack Recall | Attack Precision | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9108 | 0.1962 | 0.1088 | 0.9966 | 1.0000 | 0.9099 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8219 | 0.1974 | 0.1096 | 0.9983 | 1.0000 | 0.8179 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.7328 | 0.1975 | 0.1096 | 0.9992 | 1.0000 | 0.7238 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.6438 | 0.1975 | 0.1096 | 0.9994 | 1.0000 | 0.6275 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.5548 | 0.1975 | 0.1096 | 0.9997 | 1.0000 | 0.5290 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.4657 | 0.1975 | 0.1096 | 0.9997 | 0.9999 | 0.4281 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.3767 | 0.1975 | 0.1096 | 0.9998 | 1.0000 | 0.3249 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.2876 | 0.1975 | 0.1096 | 0.9998 | 0.9999 | 0.2192 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.1986 | 0.1975 | 0.1096 | 1.0000 | 1.0000 | 0.1109 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.1096 | 0.1975 | 0.1096 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 1.0000
- **F1-Score**: 0.0000 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.0000
- **Attack Precision** (of predicted attack, actual attack): 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9108
- **F1-Score**: 0.1962 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1088
- **Attack Precision** (of predicted attack, actual attack): 0.9966
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.9099

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8219
- **F1-Score**: 0.1974 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9983
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.8179

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.7328
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9992
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.7238

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.6438
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9994
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.6275

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.5548
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9997
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.5290

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.4657
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9997
- **Normal Recall** (of actual normal, predicted normal): 0.9999
- **Normal Precision** (of predicted normal, actual normal): 0.4281

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.3767
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9998
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.3249

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.2876
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 0.9998
- **Normal Recall** (of actual normal, predicted normal): 0.9999
- **Normal Precision** (of predicted normal, actual normal): 0.2192

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.1986
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.1109

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.1096
- **F1-Score**: 0.1975 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1096
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT

**Model path**: `models/tflite/saved_model_pruned_qat.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | F1-Score | Attack Recall | Attack Precision | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.9979 | 0.0000 | 0.0000 | 0.0000 | 0.9979 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9453 | 0.6328 | 0.4715 | 0.9620 | 0.9979 | 0.9444 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8926 | 0.6368 | 0.4710 | 0.9827 | 0.9979 | 0.8830 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.8399 | 0.6383 | 0.4711 | 0.9899 | 0.9979 | 0.8149 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.7872 | 0.6391 | 0.4710 | 0.9937 | 0.9980 | 0.7389 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.7345 | 0.6395 | 0.4710 | 0.9955 | 0.9979 | 0.6536 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.6817 | 0.6398 | 0.4710 | 0.9968 | 0.9978 | 0.5570 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.6290 | 0.6400 | 0.4710 | 0.9978 | 0.9976 | 0.4470 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.5765 | 0.6402 | 0.4710 | 0.9991 | 0.9983 | 0.3206 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.5237 | 0.6403 | 0.4710 | 0.9995 | 0.9977 | 0.1733 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.4710 | 0.6404 | 0.4710 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.9979
- **F1-Score**: 0.0000 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.0000
- **Attack Precision** (of predicted attack, actual attack): 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.9979
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9453
- **F1-Score**: 0.6328 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4715
- **Attack Precision** (of predicted attack, actual attack): 0.9620
- **Normal Recall** (of actual normal, predicted normal): 0.9979
- **Normal Precision** (of predicted normal, actual normal): 0.9444

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8926
- **F1-Score**: 0.6368 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9827
- **Normal Recall** (of actual normal, predicted normal): 0.9979
- **Normal Precision** (of predicted normal, actual normal): 0.8830

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.8399
- **F1-Score**: 0.6383 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4711
- **Attack Precision** (of predicted attack, actual attack): 0.9899
- **Normal Recall** (of actual normal, predicted normal): 0.9979
- **Normal Precision** (of predicted normal, actual normal): 0.8149

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.7872
- **F1-Score**: 0.6391 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9937
- **Normal Recall** (of actual normal, predicted normal): 0.9980
- **Normal Precision** (of predicted normal, actual normal): 0.7389

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.7345
- **F1-Score**: 0.6395 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9955
- **Normal Recall** (of actual normal, predicted normal): 0.9979
- **Normal Precision** (of predicted normal, actual normal): 0.6536

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.6817
- **F1-Score**: 0.6398 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9968
- **Normal Recall** (of actual normal, predicted normal): 0.9978
- **Normal Precision** (of predicted normal, actual normal): 0.5570

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.6290
- **F1-Score**: 0.6400 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9978
- **Normal Recall** (of actual normal, predicted normal): 0.9976
- **Normal Precision** (of predicted normal, actual normal): 0.4470

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.5765
- **F1-Score**: 0.6402 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9991
- **Normal Recall** (of actual normal, predicted normal): 0.9983
- **Normal Precision** (of predicted normal, actual normal): 0.3206

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.5237
- **F1-Score**: 0.6403 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 0.9995
- **Normal Recall** (of actual normal, predicted normal): 0.9977
- **Normal Precision** (of predicted normal, actual normal): 0.1733

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.4710
- **F1-Score**: 0.6404 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.4710
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## QAT+PTQ

**Model path**: `models/tflite/saved_model_qat_ptq.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | F1-Score | Attack Recall | Attack Precision | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9107 | 0.1934 | 0.1071 | 0.9972 | 1.0000 | 0.9097 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.8216 | 0.1949 | 0.1080 | 0.9987 | 1.0000 | 0.8177 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.7324 | 0.1949 | 0.1080 | 0.9995 | 1.0000 | 0.7234 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.6432 | 0.1949 | 0.1080 | 0.9995 | 1.0000 | 0.6271 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.5540 | 0.1949 | 0.1080 | 0.9997 | 1.0000 | 0.5285 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.4648 | 0.1949 | 0.1080 | 0.9998 | 1.0000 | 0.4277 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.3756 | 0.1949 | 0.1080 | 0.9998 | 1.0000 | 0.3245 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.2864 | 0.1949 | 0.1080 | 1.0000 | 1.0000 | 0.2189 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.1972 | 0.1949 | 0.1080 | 0.9998 | 0.9998 | 0.1107 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.1080 | 0.1949 | 0.1080 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 1.0000
- **F1-Score**: 0.0000 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.0000
- **Attack Precision** (of predicted attack, actual attack): 0.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9107
- **F1-Score**: 0.1934 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1071
- **Attack Precision** (of predicted attack, actual attack): 0.9972
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.9097

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.8216
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9987
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.8177

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.7324
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9995
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.7234

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.6432
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9995
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.6271

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.5540
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9997
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.5285

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.4648
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9998
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.4277

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.3756
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9998
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.3245

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.2864
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 1.0000
- **Normal Precision** (of predicted normal, actual normal): 0.2189

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.1972
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 0.9998
- **Normal Recall** (of actual normal, predicted normal): 0.9998
- **Normal Precision** (of predicted normal, actual normal): 0.1107

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.1080
- **F1-Score**: 0.1949 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.1080
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


---

## noQAT+PTQ

**Model path**: `models/tflite/saved_model_no_qat_ptq.tflite`

### Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | F1-Score | Attack Recall | Attack Precision | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.8923 | 0.0000 | 0.0000 | 0.0000 | 0.8923 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.9016 | 0.6676 | 0.9883 | 0.5040 | 0.8919 | 0.9985 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.9111 | 0.8163 | 0.9875 | 0.6956 | 0.8920 | 0.9965 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.9209 | 0.8822 | 0.9875 | 0.7972 | 0.8923 | 0.9940 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.9299 | 0.9185 | 0.9875 | 0.8585 | 0.8915 | 0.9907 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.9398 | 0.9425 | 0.9875 | 0.9015 | 0.8921 | 0.9862 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.9494 | 0.9590 | 0.9875 | 0.9322 | 0.8922 | 0.9794 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.9588 | 0.9710 | 0.9875 | 0.9551 | 0.8917 | 0.9683 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.9683 | 0.9803 | 0.9875 | 0.9733 | 0.8916 | 0.9469 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.9782 | 0.9879 | 0.9875 | 0.9883 | 0.8950 | 0.8882 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.9875 | 0.9937 | 0.9875 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.8923
- **F1-Score**: 0.0000 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.0000
- **Attack Precision** (of predicted attack, actual attack): 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.8923
- **Normal Precision** (of predicted normal, actual normal): 1.0000

#### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.9016
- **F1-Score**: 0.6676 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9883
- **Attack Precision** (of predicted attack, actual attack): 0.5040
- **Normal Recall** (of actual normal, predicted normal): 0.8919
- **Normal Precision** (of predicted normal, actual normal): 0.9985

#### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.9111
- **F1-Score**: 0.8163 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.6956
- **Normal Recall** (of actual normal, predicted normal): 0.8920
- **Normal Precision** (of predicted normal, actual normal): 0.9965

#### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.9209
- **F1-Score**: 0.8822 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.7972
- **Normal Recall** (of actual normal, predicted normal): 0.8923
- **Normal Precision** (of predicted normal, actual normal): 0.9940

#### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.9299
- **F1-Score**: 0.9185 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.8585
- **Normal Recall** (of actual normal, predicted normal): 0.8915
- **Normal Precision** (of predicted normal, actual normal): 0.9907

#### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.9398
- **F1-Score**: 0.9425 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.9015
- **Normal Recall** (of actual normal, predicted normal): 0.8921
- **Normal Precision** (of predicted normal, actual normal): 0.9862

#### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.9494
- **F1-Score**: 0.9590 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.9322
- **Normal Recall** (of actual normal, predicted normal): 0.8922
- **Normal Precision** (of predicted normal, actual normal): 0.9794

#### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.9588
- **F1-Score**: 0.9710 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.9551
- **Normal Recall** (of actual normal, predicted normal): 0.8917
- **Normal Precision** (of predicted normal, actual normal): 0.9683

#### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.9683
- **F1-Score**: 0.9803 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.9733
- **Normal Recall** (of actual normal, predicted normal): 0.8916
- **Normal Precision** (of predicted normal, actual normal): 0.9469

#### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.9782
- **F1-Score**: 0.9879 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 0.9883
- **Normal Recall** (of actual normal, predicted normal): 0.8950
- **Normal Precision** (of predicted normal, actual normal): 0.8882

#### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.9875
- **F1-Score**: 0.9937 (from Attack Recall & Attack Precision)
- **Attack Recall** (of actual attack, predicted attack): 0.9875
- **Attack Precision** (of predicted attack, actual attack): 1.0000
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000



## Threshold Tuning (best per ratio)
Metric used for recommendation: **f1**

### Original (TFLite)

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.1648 | 0.0000 | 0.1648 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.30 | 0.9287 | 0.5517 | 0.9831 | 0.9404 | 0.4390 | 0.7423 |
| 80 | 20 | 0.20 | 0.8622 | 0.5910 | 0.9533 | 0.8836 | 0.4978 | 0.7273 |
| 70 | 30 | 0.20 | 0.8161 | 0.6190 | 0.9526 | 0.8157 | 0.4978 | 0.8181 |
| 60 | 40 | 0.20 | 0.7713 | 0.6352 | 0.9536 | 0.7401 | 0.4978 | 0.8774 |
| 50 | 50 | 0.15 | 0.5462 | 0.6709 | 0.1672 | 0.6907 | 0.9251 | 0.5263 |
| 40 | 60 | 0.15 | 0.6207 | 0.7453 | 0.1640 | 0.5934 | 0.9251 | 0.6240 |
| 30 | 70 | 0.15 | 0.6979 | 0.8108 | 0.1676 | 0.4896 | 0.9251 | 0.7217 |
| 20 | 80 | 0.15 | 0.7731 | 0.8671 | 0.1649 | 0.3550 | 0.9251 | 0.8159 |
| 10 | 90 | 0.15 | 0.8494 | 0.9171 | 0.1685 | 0.2000 | 0.9251 | 0.9092 |
| 0 | 100 | 0.15 | 0.9251 | 0.9611 | 0.0000 | 0.0000 | 0.9251 | 1.0000 |

### QAT+Prune only

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.0449 | 0.0000 | 0.0449 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.25 | 0.9208 | 0.4444 | 0.9879 | 0.9286 | 0.3168 | 0.7441 |
| 80 | 20 | 0.20 | 0.8476 | 0.4780 | 0.9722 | 0.8566 | 0.3490 | 0.7586 |
| 70 | 30 | 0.20 | 0.7851 | 0.4935 | 0.9720 | 0.7770 | 0.3490 | 0.8425 |
| 60 | 40 | 0.15 | 0.4265 | 0.5823 | 0.0445 | 0.9913 | 0.9994 | 0.4108 |
| 50 | 50 | 0.15 | 0.5229 | 0.6769 | 0.0464 | 0.9876 | 0.9994 | 0.5117 |
| 40 | 60 | 0.15 | 0.6174 | 0.7582 | 0.0445 | 0.9807 | 0.9994 | 0.6107 |
| 30 | 70 | 0.15 | 0.7130 | 0.8298 | 0.0449 | 0.9705 | 0.9994 | 0.7094 |
| 20 | 80 | 0.15 | 0.8088 | 0.8932 | 0.0461 | 0.9518 | 0.9994 | 0.8074 |
| 10 | 90 | 0.15 | 0.9040 | 0.9493 | 0.0451 | 0.8957 | 0.9994 | 0.9040 |
| 0 | 100 | 0.15 | 0.9994 | 0.9997 | 0.0000 | 0.0000 | 0.9994 | 1.0000 |

### QAT

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.5229 | 0.0000 | 0.5229 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.45 | 0.9452 | 0.6331 | 0.9978 | 0.9445 | 0.4724 | 0.9595 |
| 80 | 20 | 0.25 | 0.8890 | 0.6386 | 0.9886 | 0.8858 | 0.4904 | 0.9149 |
| 70 | 30 | 0.25 | 0.8391 | 0.6464 | 0.9885 | 0.8190 | 0.4904 | 0.9480 |
| 60 | 40 | 0.15 | 0.6974 | 0.7167 | 0.5243 | 0.9482 | 0.9571 | 0.5729 |
| 50 | 50 | 0.15 | 0.7418 | 0.7875 | 0.5265 | 0.9246 | 0.9571 | 0.6690 |
| 40 | 60 | 0.15 | 0.7833 | 0.8413 | 0.5228 | 0.8903 | 0.9571 | 0.7505 |
| 30 | 70 | 0.15 | 0.8277 | 0.8860 | 0.5257 | 0.8400 | 0.9571 | 0.8248 |
| 20 | 80 | 0.15 | 0.8712 | 0.9224 | 0.5279 | 0.7546 | 0.9571 | 0.8902 |
| 10 | 90 | 0.15 | 0.9142 | 0.9526 | 0.5283 | 0.5776 | 0.9571 | 0.9481 |
| 0 | 100 | 0.15 | 0.9571 | 0.9781 | 0.0000 | 0.0000 | 0.9571 | 1.0000 |

### QAT+PTQ

| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |
|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|
| 100 | 0 | 0.15 | 0.0451 | 0.0000 | 0.0451 | 1.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 0.25 | 0.9204 | 0.4382 | 0.9881 | 0.9281 | 0.3106 | 0.7440 |
| 80 | 20 | 0.20 | 0.8475 | 0.4765 | 0.9726 | 0.8563 | 0.3471 | 0.7600 |
| 70 | 30 | 0.20 | 0.7846 | 0.4915 | 0.9721 | 0.7765 | 0.3471 | 0.8421 |
| 60 | 40 | 0.15 | 0.4266 | 0.5823 | 0.0447 | 0.9904 | 0.9993 | 0.4109 |
| 50 | 50 | 0.15 | 0.5221 | 0.6765 | 0.0449 | 0.9857 | 0.9993 | 0.5113 |
| 40 | 60 | 0.15 | 0.6174 | 0.7581 | 0.0444 | 0.9784 | 0.9993 | 0.6107 |
| 30 | 70 | 0.15 | 0.7132 | 0.8299 | 0.0455 | 0.9677 | 0.9993 | 0.7096 |
| 20 | 80 | 0.15 | 0.8085 | 0.8930 | 0.0450 | 0.9452 | 0.9993 | 0.8072 |
| 10 | 90 | 0.15 | 0.9036 | 0.9492 | 0.0423 | 0.8782 | 0.9993 | 0.9038 |
| 0 | 100 | 0.15 | 0.9993 | 0.9997 | 0.0000 | 0.0000 | 0.9993 | 1.0000 |

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

