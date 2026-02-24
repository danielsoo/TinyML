# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-24 10:17:52 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0158 | 0.1130 | 0.2107 | 0.3085 | 0.4060 | 0.5038 | 0.6014 | 0.6993 | 0.7973 | 0.8950 | 0.9926 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| saved_model_traditional_qat | 0.9685 | 0.9663 | 0.9639 | 0.9618 | 0.9593 | 0.9566 | 0.9545 | 0.9516 | 0.9493 | 0.9468 | 0.9444 |
| QAT+PTQ | 0.0009 | 0.1009 | 0.2007 | 0.3007 | 0.4004 | 0.5001 | 0.6001 | 0.6998 | 0.7996 | 0.8994 | 0.9993 |
| Compressed (QAT) | 0.9711 | 0.9662 | 0.9606 | 0.9555 | 0.9498 | 0.9445 | 0.9392 | 0.9334 | 0.9283 | 0.9228 | 0.9174 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1828 | 0.3347 | 0.4627 | 0.5721 | 0.6667 | 0.7493 | 0.8221 | 0.8868 | 0.9445 | 0.9963 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8488 | 0.9128 | 0.9368 | 0.9489 | 0.9561 | 0.9614 | 0.9647 | 0.9676 | 0.9696 | 0.9714 |
| QAT+PTQ | 0.0000 | 0.1819 | 0.3334 | 0.4616 | 0.5714 | 0.6666 | 0.7499 | 0.8233 | 0.8886 | 0.9470 | 0.9996 |
| Compressed (QAT) | 0.0000 | 0.8447 | 0.9031 | 0.9253 | 0.9359 | 0.9430 | 0.9477 | 0.9507 | 0.9534 | 0.9554 | 0.9569 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0158 | 0.0153 | 0.0153 | 0.0153 | 0.0149 | 0.0151 | 0.0146 | 0.0148 | 0.0159 | 0.0162 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| saved_model_traditional_qat | 0.9685 | 0.9688 | 0.9688 | 0.9692 | 0.9692 | 0.9688 | 0.9695 | 0.9682 | 0.9688 | 0.9679 | 0.0000 |
| QAT+PTQ | 0.0009 | 0.0011 | 0.0011 | 0.0013 | 0.0012 | 0.0010 | 0.0013 | 0.0011 | 0.0009 | 0.0002 | 0.0000 |
| Compressed (QAT) | 0.9711 | 0.9715 | 0.9714 | 0.9719 | 0.9713 | 0.9716 | 0.9719 | 0.9708 | 0.9716 | 0.9714 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0158 | 0.0000 | 0.0000 | 0.0000 | 0.0158 | 1.0000 |
| 90 | 10 | 299,940 | 0.1130 | 0.1007 | 0.9923 | 0.1828 | 0.0153 | 0.9469 |
| 80 | 20 | 291,350 | 0.2107 | 0.2013 | 0.9926 | 0.3347 | 0.0153 | 0.8919 |
| 70 | 30 | 194,230 | 0.3085 | 0.3017 | 0.9926 | 0.4627 | 0.0153 | 0.8288 |
| 60 | 40 | 145,675 | 0.4060 | 0.4018 | 0.9926 | 0.5721 | 0.0149 | 0.7510 |
| 50 | 50 | 116,540 | 0.5038 | 0.5019 | 0.9926 | 0.6667 | 0.0151 | 0.6705 |
| 40 | 60 | 97,115 | 0.6014 | 0.6017 | 0.9926 | 0.7493 | 0.0146 | 0.5681 |
| 30 | 70 | 83,240 | 0.6993 | 0.7016 | 0.9926 | 0.8221 | 0.0148 | 0.4619 |
| 20 | 80 | 72,835 | 0.7973 | 0.8014 | 0.9926 | 0.8868 | 0.0159 | 0.3489 |
| 10 | 90 | 64,740 | 0.8950 | 0.9008 | 0.9926 | 0.9445 | 0.0162 | 0.1959 |
| 0 | 100 | 58,270 | 0.9926 | 1.0000 | 0.9926 | 0.9963 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0521 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 1.0000 |
| 90 | 10 | 299,940 | 0.1469 | 0.1049 | 1.0000 | 0.1899 | 0.0521 | 1.0000 |
| 80 | 20 | 291,350 | 0.2418 | 0.2087 | 1.0000 | 0.3454 | 0.0522 | 1.0000 |
| 70 | 30 | 194,230 | 0.3362 | 0.3113 | 1.0000 | 0.4748 | 0.0518 | 1.0000 |
| 60 | 40 | 145,675 | 0.4319 | 0.4132 | 1.0000 | 0.5847 | 0.0531 | 1.0000 |
| 50 | 50 | 116,540 | 0.5260 | 0.5133 | 1.0000 | 0.6784 | 0.0519 | 1.0000 |
| 40 | 60 | 97,115 | 0.6211 | 0.6129 | 1.0000 | 0.7600 | 0.0527 | 1.0000 |
| 30 | 70 | 83,240 | 0.7160 | 0.7114 | 1.0000 | 0.8314 | 0.0535 | 1.0000 |
| 20 | 80 | 72,835 | 0.8101 | 0.8082 | 1.0000 | 0.8939 | 0.0505 | 1.0000 |
| 10 | 90 | 64,740 | 0.9056 | 0.9051 | 1.0000 | 0.9502 | 0.0562 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### saved_model_traditional_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9685 | 0.0000 | 0.0000 | 0.0000 | 0.9685 | 1.0000 |
| 90 | 10 | 299,940 | 0.9663 | 0.7706 | 0.9446 | 0.8488 | 0.9688 | 0.9937 |
| 80 | 20 | 291,350 | 0.9639 | 0.8831 | 0.9444 | 0.9128 | 0.9688 | 0.9859 |
| 70 | 30 | 194,230 | 0.9618 | 0.9293 | 0.9444 | 0.9368 | 0.9692 | 0.9760 |
| 60 | 40 | 145,675 | 0.9593 | 0.9533 | 0.9444 | 0.9489 | 0.9692 | 0.9632 |
| 50 | 50 | 116,540 | 0.9566 | 0.9680 | 0.9444 | 0.9561 | 0.9688 | 0.9458 |
| 40 | 60 | 97,115 | 0.9545 | 0.9789 | 0.9444 | 0.9614 | 0.9695 | 0.9209 |
| 30 | 70 | 83,240 | 0.9516 | 0.9858 | 0.9444 | 0.9647 | 0.9682 | 0.8819 |
| 20 | 80 | 72,835 | 0.9493 | 0.9918 | 0.9445 | 0.9676 | 0.9688 | 0.8135 |
| 10 | 90 | 64,740 | 0.9468 | 0.9962 | 0.9444 | 0.9696 | 0.9679 | 0.6594 |
| 0 | 100 | 58,270 | 0.9444 | 1.0000 | 0.9444 | 0.9714 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 1.0000 |
| 90 | 10 | 299,940 | 0.1009 | 0.1000 | 0.9994 | 0.1819 | 0.0011 | 0.9403 |
| 80 | 20 | 291,350 | 0.2007 | 0.2001 | 0.9993 | 0.3334 | 0.0011 | 0.8562 |
| 70 | 30 | 194,230 | 0.3007 | 0.3001 | 0.9993 | 0.4616 | 0.0013 | 0.7981 |
| 60 | 40 | 145,675 | 0.4004 | 0.4001 | 0.9993 | 0.5714 | 0.0012 | 0.7114 |
| 50 | 50 | 116,540 | 0.5001 | 0.5001 | 0.9993 | 0.6666 | 0.0010 | 0.5743 |
| 40 | 60 | 97,115 | 0.6001 | 0.6001 | 0.9993 | 0.7499 | 0.0013 | 0.5426 |
| 30 | 70 | 83,240 | 0.6998 | 0.7001 | 0.9993 | 0.8233 | 0.0011 | 0.3857 |
| 20 | 80 | 72,835 | 0.7996 | 0.8000 | 0.9993 | 0.8886 | 0.0009 | 0.2321 |
| 10 | 90 | 64,740 | 0.8994 | 0.8999 | 0.9993 | 0.9470 | 0.0002 | 0.0227 |
| 0 | 100 | 58,270 | 0.9993 | 1.0000 | 0.9993 | 0.9996 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9711 | 0.0000 | 0.0000 | 0.0000 | 0.9711 | 1.0000 |
| 90 | 10 | 299,940 | 0.9662 | 0.7818 | 0.9185 | 0.8447 | 0.9715 | 0.9908 |
| 80 | 20 | 291,350 | 0.9606 | 0.8891 | 0.9174 | 0.9031 | 0.9714 | 0.9792 |
| 70 | 30 | 194,230 | 0.9555 | 0.9332 | 0.9174 | 0.9253 | 0.9719 | 0.9649 |
| 60 | 40 | 145,675 | 0.9498 | 0.9552 | 0.9174 | 0.9359 | 0.9713 | 0.9464 |
| 50 | 50 | 116,540 | 0.9445 | 0.9700 | 0.9174 | 0.9430 | 0.9716 | 0.9217 |
| 40 | 60 | 97,115 | 0.9392 | 0.9800 | 0.9174 | 0.9477 | 0.9719 | 0.8870 |
| 30 | 70 | 83,240 | 0.9334 | 0.9865 | 0.9174 | 0.9507 | 0.9708 | 0.8344 |
| 20 | 80 | 72,835 | 0.9283 | 0.9923 | 0.9175 | 0.9534 | 0.9716 | 0.7463 |
| 10 | 90 | 64,740 | 0.9228 | 0.9966 | 0.9174 | 0.9554 | 0.9714 | 0.5666 |
| 0 | 100 | 58,270 | 0.9174 | 1.0000 | 0.9174 | 0.9569 | 0.0000 | 0.0000 |


## Threshold Tuning (Original)

Model: `models/tflite/saved_model_original.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1010   0.1817   0.0013   0.8445   0.9978   0.0999  
0.20       0.1038   0.1821   0.0045   0.9407   0.9975   0.1002  
0.25       0.1064   0.1822   0.0076   0.9408   0.9957   0.1003  
0.30       0.1129   0.1828   0.0153   0.9456   0.9921   0.1007  
0.35       0.1176   0.1833   0.0206   0.9489   0.9900   0.1010   <--
0.40       0.1187   0.1691   0.0322   0.7379   0.8971   0.0934  
0.45       0.1625   0.1099   0.1231   0.6965   0.5172   0.0615  
0.50       0.8617   0.1274   0.9462   0.9045   0.1010   0.1726  
0.55       0.8991   0.0002   0.9989   0.8999   0.0001   0.0103  
0.60       0.8997   0.0000   0.9997   0.9000   0.0000   0.0000  
0.65       0.8999   0.0000   0.9999   0.9000   0.0000   0.0000  
0.70       0.8999   0.0000   0.9999   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.1176, F1=0.1833, Normal Recall=0.0206, Normal Precision=0.9489, Attack Recall=0.9900, Attack Precision=0.1010

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2006   0.3330   0.0013   0.7150   0.9979   0.1999  
0.20       0.2031   0.3337   0.0045   0.8815   0.9976   0.2003  
0.25       0.2053   0.3339   0.0077   0.8798   0.9958   0.2006  
0.30       0.2109   0.3347   0.0154   0.8929   0.9926   0.2013  
0.35       0.2148   0.3354   0.0209   0.8981   0.9905   0.2019   <--
0.40       0.2056   0.3117   0.0323   0.5613   0.8992   0.1885  
0.45       0.2019   0.2058   0.1231   0.5048   0.5172   0.1285  
0.50       0.7770   0.1505   0.9466   0.8077   0.0988   0.3161  
0.55       0.7992   0.0003   0.9990   0.7999   0.0001   0.0319  
0.60       0.7998   0.0001   0.9997   0.8000   0.0001   0.0429  
0.65       0.7999   0.0000   0.9999   0.8000   0.0000   0.0370  
0.70       0.8000   0.0000   0.9999   0.8000   0.0000   0.0714  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.1111  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.2148, F1=0.3354, Normal Recall=0.0209, Normal Precision=0.8981, Attack Recall=0.9905, Attack Precision=0.2019

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3003   0.4611   0.0014   0.6078   0.9979   0.2999  
0.20       0.3024   0.4618   0.0045   0.8112   0.9976   0.3004  
0.25       0.3041   0.4620   0.0077   0.8103   0.9958   0.3007  
0.30       0.3087   0.4628   0.0156   0.8312   0.9926   0.3017  
0.35       0.3118   0.4634   0.0209   0.8371   0.9905   0.3024   <--
0.40       0.2927   0.4327   0.0327   0.4310   0.8992   0.2849  
0.45       0.2422   0.2905   0.1243   0.3753   0.5172   0.2020  
0.50       0.6928   0.1618   0.9474   0.7104   0.0988   0.4460  
0.55       0.6993   0.0003   0.9989   0.6998   0.0001   0.0516  
0.60       0.6998   0.0001   0.9997   0.7000   0.0001   0.0714  
0.65       0.6999   0.0000   0.9999   0.7000   0.0000   0.0625  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.1429  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.2500  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.3118, F1=0.4634, Normal Recall=0.0209, Normal Precision=0.8371, Attack Recall=0.9905, Attack Precision=0.3024

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4000   0.5709   0.0014   0.5122   0.9979   0.3999  
0.20       0.4018   0.5716   0.0046   0.7409   0.9976   0.4005  
0.25       0.4030   0.5716   0.0078   0.7356   0.9958   0.4009  
0.30       0.4065   0.5723   0.0158   0.7619   0.9926   0.4020  
0.35       0.4087   0.5727   0.0209   0.7676   0.9905   0.4028   <--
0.40       0.3793   0.5368   0.0328   0.3276   0.8992   0.3826  
0.45       0.2811   0.3653   0.1237   0.2777   0.5172   0.2824  
0.50       0.6076   0.1677   0.9469   0.6118   0.0988   0.5536  
0.55       0.5994   0.0003   0.9989   0.5998   0.0001   0.0769  
0.60       0.5999   0.0001   0.9997   0.5999   0.0001   0.1154  
0.65       0.6000   0.0000   0.9999   0.6000   0.0000   0.1111  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.2000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.3333  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.4087, F1=0.5727, Normal Recall=0.0209, Normal Precision=0.7676, Attack Recall=0.9905, Attack Precision=0.4028

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4997   0.6661   0.0015   0.4175   0.9979   0.4999  
0.20       0.5011   0.6666   0.0047   0.6562   0.9976   0.5006  
0.25       0.5017   0.6665   0.0077   0.6469   0.9958   0.5009  
0.30       0.5042   0.6669   0.0157   0.6803   0.9926   0.5021  
0.35       0.5056   0.6671   0.0207   0.6856   0.9905   0.5028   <--
0.40       0.4658   0.6273   0.0325   0.2435   0.8992   0.4817  
0.45       0.3203   0.4321   0.1234   0.2036   0.5172   0.3711  
0.50       0.5223   0.1714   0.9458   0.5121   0.0988   0.6459  
0.55       0.4995   0.0003   0.9990   0.4998   0.0001   0.1159  
0.60       0.4999   0.0001   0.9998   0.5000   0.0001   0.1765  
0.65       0.5000   0.0000   0.9999   0.5000   0.0000   0.2000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.3333  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.3333  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.5056, F1=0.6671, Normal Recall=0.0207, Normal Precision=0.6856, Attack Recall=0.9905, Attack Precision=0.5028

```


## Threshold Tuning (noQAT+PTQ)

Model: `models/tflite/saved_model_no_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1129   0.1840   0.0143   1.0000   1.0000   0.1013  
0.20       0.1187   0.1850   0.0208   1.0000   1.0000   0.1019  
0.25       0.1314   0.1872   0.0348   1.0000   1.0000   0.1032  
0.30       0.1469   0.1899   0.0521   1.0000   1.0000   0.1049  
0.35       0.1772   0.1955   0.0858   1.0000   1.0000   0.1084  
0.40       0.2735   0.2159   0.1928   1.0000   1.0000   0.1210  
0.45       0.5748   0.3199   0.5276   1.0000   0.9998   0.1904  
0.50       0.7845   0.4806   0.7608   0.9996   0.9974   0.3166  
0.55       0.9485   0.7618   0.9624   0.9800   0.8235   0.7087   <--
0.60       0.9503   0.7017   0.9911   0.9554   0.5840   0.8789  
0.65       0.9471   0.6485   0.9982   0.9461   0.4878   0.9671  
0.70       0.9475   0.6466   0.9993   0.9454   0.4805   0.9879  
0.75       0.9465   0.6357   0.9997   0.9441   0.4671   0.9945  
0.80       0.9413   0.5858   0.9998   0.9389   0.4148   0.9967  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9485, F1=0.7618, Normal Recall=0.9624, Normal Precision=0.9800, Attack Recall=0.8235, Attack Precision=0.7087

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2113   0.3365   0.0142   1.0000   1.0000   0.2023  
0.20       0.2166   0.3380   0.0207   1.0000   1.0000   0.2034  
0.25       0.2278   0.3412   0.0347   1.0000   1.0000   0.2057  
0.30       0.2415   0.3453   0.0518   1.0000   1.0000   0.2087  
0.35       0.2684   0.3535   0.0855   0.9999   1.0000   0.2147  
0.40       0.3541   0.3824   0.1927   1.0000   1.0000   0.2364  
0.45       0.6221   0.5141   0.5277   0.9999   0.9997   0.3460  
0.50       0.8083   0.6754   0.7610   0.9991   0.9972   0.5106  
0.55       0.9345   0.8342   0.9623   0.9561   0.8233   0.8453   <--
0.60       0.9099   0.7220   0.9912   0.9052   0.5849   0.9431  
0.65       0.8961   0.6524   0.9982   0.8863   0.4876   0.9852  
0.70       0.8955   0.6478   0.9993   0.8850   0.4803   0.9945  
0.75       0.8931   0.6359   0.9997   0.8823   0.4667   0.9975  
0.80       0.8826   0.5850   0.9998   0.8721   0.4137   0.9985  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9345, F1=0.8342, Normal Recall=0.9623, Normal Precision=0.9561, Attack Recall=0.8233, Attack Precision=0.8453

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3100   0.4651   0.0143   1.0000   1.0000   0.3030  
0.20       0.3149   0.4669   0.0213   1.0000   1.0000   0.3045  
0.25       0.3248   0.4705   0.0354   1.0000   1.0000   0.3076  
0.30       0.3369   0.4750   0.0528   1.0000   1.0000   0.3115  
0.35       0.3605   0.4841   0.0865   0.9999   1.0000   0.3193  
0.40       0.4358   0.5154   0.1940   0.9999   1.0000   0.3471  
0.45       0.6702   0.6452   0.5290   0.9998   0.9997   0.4763  
0.50       0.8314   0.7802   0.7604   0.9984   0.9972   0.6408  
0.55       0.9203   0.8611   0.9619   0.9270   0.8233   0.9025   <--
0.60       0.8691   0.7283   0.9909   0.8478   0.5849   0.9648  
0.65       0.8450   0.6537   0.9981   0.8197   0.4876   0.9911  
0.70       0.8436   0.6483   0.9993   0.8178   0.4803   0.9967  
0.75       0.8398   0.6361   0.9997   0.8139   0.4667   0.9986  
0.80       0.8240   0.5851   0.9999   0.7992   0.4137   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9203, F1=0.8611, Normal Recall=0.9619, Normal Precision=0.9270, Attack Recall=0.8233, Attack Precision=0.9025

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4083   0.5748   0.0138   1.0000   1.0000   0.4033  
0.20       0.4125   0.5766   0.0208   1.0000   1.0000   0.4051  
0.25       0.4211   0.5802   0.0351   1.0000   1.0000   0.4086  
0.30       0.4314   0.5845   0.0523   1.0000   1.0000   0.4129  
0.35       0.4518   0.5934   0.0863   0.9999   1.0000   0.4218  
0.40       0.5165   0.6233   0.1942   0.9999   1.0000   0.4527  
0.45       0.7172   0.7388   0.5288   0.9996   0.9997   0.5858  
0.50       0.8550   0.8462   0.7603   0.9975   0.9972   0.7350  
0.55       0.9065   0.8757   0.9619   0.8909   0.8233   0.9351   <--
0.60       0.8287   0.7320   0.9912   0.7817   0.5849   0.9780  
0.65       0.7940   0.6544   0.9982   0.7451   0.4876   0.9946  
0.70       0.7918   0.6485   0.9994   0.7426   0.4803   0.9981  
0.75       0.7865   0.6362   0.9997   0.7376   0.4667   0.9990  
0.80       0.7654   0.5852   0.9998   0.7189   0.4137   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9065, F1=0.8757, Normal Recall=0.9619, Normal Precision=0.8909, Attack Recall=0.8233, Attack Precision=0.9351

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5069   0.6697   0.0137   1.0000   1.0000   0.5035  
0.20       0.5103   0.6713   0.0205   1.0000   1.0000   0.5052  
0.25       0.5173   0.6745   0.0346   1.0000   1.0000   0.5088  
0.30       0.5259   0.6784   0.0517   1.0000   1.0000   0.5133  
0.35       0.5430   0.6863   0.0860   0.9998   1.0000   0.5225  
0.40       0.5965   0.7125   0.1930   0.9998   1.0000   0.5534  
0.45       0.7643   0.8092   0.5289   0.9994   0.9997   0.6797  
0.50       0.8788   0.8917   0.7605   0.9963   0.9972   0.8063   <--
0.55       0.8926   0.8846   0.9618   0.8448   0.8233   0.9556  
0.60       0.7880   0.7340   0.9911   0.7048   0.5849   0.9851  
0.65       0.7429   0.6548   0.9982   0.6608   0.4876   0.9964  
0.70       0.7399   0.6487   0.9994   0.6579   0.4803   0.9987  
0.75       0.7332   0.6362   0.9997   0.6521   0.4667   0.9993  
0.80       0.7068   0.5852   0.9998   0.6304   0.4137   0.9996  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8788, F1=0.8917, Normal Recall=0.7605, Normal Precision=0.9963, Attack Recall=0.9972, Attack Precision=0.8063

```


## Threshold Tuning (saved_model_traditional_qat)

Model: `models/tflite/saved_model_traditional_qat.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9445   0.7780   0.9415   0.9967   0.9721   0.6485  
0.20       0.9610   0.8301   0.9620   0.9945   0.9519   0.7359  
0.25       0.9649   0.8435   0.9670   0.9938   0.9458   0.7612  
0.30       0.9664   0.8492   0.9688   0.9938   0.9454   0.7708  
0.35       0.9679   0.8537   0.9713   0.9929   0.9373   0.7839  
0.40       0.9688   0.8566   0.9729   0.9923   0.9319   0.7926  
0.45       0.9698   0.8605   0.9741   0.9922   0.9313   0.7997  
0.50       0.9701   0.8618   0.9745   0.9922   0.9309   0.8022  
0.55       0.9697   0.8589   0.9749   0.9912   0.9223   0.8036  
0.60       0.9737   0.8623   0.9905   0.9805   0.8225   0.9062  
0.65       0.9755   0.8682   0.9944   0.9788   0.8060   0.9408  
0.70       0.9760   0.8689   0.9960   0.9777   0.7957   0.9569   <--
0.75       0.9753   0.8636   0.9970   0.9761   0.7804   0.9666  
0.80       0.9757   0.8650   0.9974   0.9760   0.7797   0.9713  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9760, F1=0.8689, Normal Recall=0.9960, Normal Precision=0.9777, Attack Recall=0.7957, Attack Precision=0.9569

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9473   0.8805   0.9414   0.9923   0.9709   0.8056  
0.20       0.9599   0.9046   0.9621   0.9874   0.9510   0.8625  
0.25       0.9626   0.9100   0.9671   0.9859   0.9447   0.8777  
0.30       0.9640   0.9129   0.9688   0.9859   0.9444   0.8834  
0.35       0.9642   0.9128   0.9713   0.9837   0.9358   0.8909  
0.40       0.9645   0.9129   0.9730   0.9825   0.9306   0.8959  
0.45       0.9653   0.9146   0.9742   0.9823   0.9298   0.9000  
0.50       0.9655   0.9151   0.9745   0.9822   0.9294   0.9013   <--
0.55       0.9641   0.9111   0.9750   0.9800   0.9205   0.9019  
0.60       0.9564   0.8827   0.9906   0.9565   0.8198   0.9562  
0.65       0.9563   0.8805   0.9944   0.9531   0.8042   0.9728  
0.70       0.9556   0.8773   0.9960   0.9508   0.7939   0.9803  
0.75       0.9535   0.8702   0.9970   0.9476   0.7794   0.9849  
0.80       0.9537   0.8705   0.9974   0.9474   0.7786   0.9871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9655, F1=0.9151, Normal Recall=0.9745, Normal Precision=0.9822, Attack Recall=0.9294, Attack Precision=0.9013

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9504   0.9215   0.9416   0.9869   0.9709   0.8769  
0.20       0.9586   0.9324   0.9619   0.9786   0.9510   0.9144  
0.25       0.9602   0.9344   0.9669   0.9761   0.9447   0.9244  
0.30       0.9614   0.9362   0.9686   0.9760   0.9444   0.9281   <--
0.35       0.9606   0.9344   0.9712   0.9724   0.9358   0.9329  
0.40       0.9601   0.9333   0.9727   0.9703   0.9306   0.9360  
0.45       0.9606   0.9341   0.9739   0.9700   0.9298   0.9385  
0.50       0.9608   0.9343   0.9742   0.9699   0.9294   0.9393  
0.55       0.9584   0.9300   0.9747   0.9662   0.9205   0.9397  
0.60       0.9394   0.8903   0.9906   0.9277   0.8197   0.9740  
0.65       0.9373   0.8850   0.9944   0.9222   0.8041   0.9839  
0.70       0.9353   0.8804   0.9959   0.9185   0.7939   0.9882  
0.75       0.9317   0.8725   0.9969   0.9134   0.7794   0.9908  
0.80       0.9318   0.8725   0.9974   0.9131   0.7786   0.9923  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9614, F1=0.9362, Normal Recall=0.9686, Normal Precision=0.9760, Attack Recall=0.9444, Attack Precision=0.9281

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9532   0.9431   0.9414   0.9798   0.9709   0.9169  
0.20       0.9575   0.9471   0.9618   0.9671   0.9510   0.9432  
0.25       0.9581   0.9474   0.9669   0.9633   0.9447   0.9501  
0.30       0.9590   0.9486   0.9687   0.9632   0.9444   0.9527   <--
0.35       0.9570   0.9457   0.9712   0.9578   0.9358   0.9558  
0.40       0.9558   0.9440   0.9727   0.9546   0.9306   0.9578  
0.45       0.9562   0.9444   0.9738   0.9541   0.9298   0.9595  
0.50       0.9563   0.9444   0.9742   0.9539   0.9294   0.9600  
0.55       0.9529   0.9399   0.9745   0.9484   0.9205   0.9601  
0.60       0.9223   0.8940   0.9906   0.8918   0.8198   0.9831  
0.65       0.9182   0.8872   0.9942   0.8839   0.8042   0.9894  
0.70       0.9150   0.8820   0.9958   0.8787   0.7939   0.9921  
0.75       0.9098   0.8736   0.9967   0.8714   0.7794   0.9937  
0.80       0.9098   0.8735   0.9972   0.8711   0.7786   0.9947  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9590, F1=0.9486, Normal Recall=0.9687, Normal Precision=0.9632, Attack Recall=0.9444, Attack Precision=0.9527

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9563   0.9569   0.9417   0.9700   0.9709   0.9433   <--
0.20       0.9566   0.9564   0.9623   0.9515   0.9510   0.9618  
0.25       0.9561   0.9556   0.9675   0.9460   0.9447   0.9667  
0.30       0.9568   0.9563   0.9692   0.9458   0.9444   0.9684  
0.35       0.9536   0.9528   0.9715   0.9380   0.9358   0.9705  
0.40       0.9518   0.9508   0.9730   0.9334   0.9306   0.9718  
0.45       0.9519   0.9508   0.9741   0.9327   0.9298   0.9729  
0.50       0.9519   0.9508   0.9744   0.9325   0.9294   0.9732  
0.55       0.9476   0.9462   0.9748   0.9246   0.9205   0.9733  
0.60       0.9052   0.8963   0.9906   0.8461   0.8198   0.9887  
0.65       0.8991   0.8885   0.9941   0.8354   0.8042   0.9927  
0.70       0.8948   0.8830   0.9957   0.8285   0.7939   0.9946  
0.75       0.8880   0.8744   0.9966   0.8188   0.7794   0.9956  
0.80       0.8879   0.8741   0.9972   0.8183   0.7786   0.9964  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9563, F1=0.9569, Normal Recall=0.9417, Normal Precision=0.9700, Attack Recall=0.9709, Attack Precision=0.9433

```


## Threshold Tuning (QAT+PTQ)

Model: `models/tflite/saved_model_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1004   0.1819   0.0004   1.0000   1.0000   0.1000  
0.20       0.1005   0.1819   0.0006   1.0000   1.0000   0.1001  
0.25       0.1009   0.1819   0.0010   0.9850   0.9999   0.1001  
0.30       0.1009   0.1819   0.0011   0.9462   0.9994   0.1000  
0.35       0.1027   0.1821   0.0031   0.9668   0.9990   0.1002  
0.40       0.1094   0.1825   0.0111   0.9426   0.9939   0.1005   <--
0.45       0.1397   0.1816   0.0491   0.9065   0.9544   0.1003  
0.50       0.8833   0.0001   0.9814   0.8983   0.0000   0.0002  
0.55       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.60       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.65       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.70       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.1094, F1=0.1825, Normal Recall=0.0111, Normal Precision=0.9426, Attack Recall=0.9939, Attack Precision=0.1005

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2003   0.3334   0.0004   0.9806   1.0000   0.2001  
0.20       0.2004   0.3334   0.0006   0.9851   1.0000   0.2001  
0.25       0.2008   0.3335   0.0010   0.9433   0.9998   0.2001  
0.30       0.2008   0.3334   0.0011   0.8599   0.9993   0.2001  
0.35       0.2023   0.3337   0.0032   0.9194   0.9989   0.2003  
0.40       0.2077   0.3342   0.0111   0.8839   0.9942   0.2009   <--
0.45       0.2302   0.3314   0.0493   0.8105   0.9539   0.2005  
0.50       0.7851   0.0001   0.9814   0.7970   0.0001   0.0009  
0.55       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.60       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.65       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.70       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.2077, F1=0.3342, Normal Recall=0.0111, Normal Precision=0.8839, Attack Recall=0.9942, Attack Precision=0.2009

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3003   0.4616   0.0004   0.9655   1.0000   0.3001  
0.20       0.3004   0.4617   0.0005   0.9730   1.0000   0.3001  
0.25       0.3006   0.4617   0.0009   0.8971   0.9998   0.3001  
0.30       0.3005   0.4615   0.0010   0.7598   0.9993   0.3001  
0.35       0.3018   0.4619   0.0030   0.8643   0.9989   0.3004  
0.40       0.3060   0.4622   0.0110   0.8149   0.9942   0.3011   <--
0.45       0.3209   0.4574   0.0497   0.7155   0.9539   0.3008  
0.50       0.6872   0.0001   0.9817   0.6961   0.0001   0.0016  
0.55       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.60       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.65       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.3060, F1=0.4622, Normal Recall=0.0110, Normal Precision=0.8149, Attack Recall=0.9942, Attack Precision=0.3011

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4002   0.5715   0.0004   0.9459   1.0000   0.4001  
0.20       0.4003   0.5715   0.0005   0.9574   1.0000   0.4001  
0.25       0.4004   0.5715   0.0008   0.8333   0.9998   0.4001  
0.30       0.4002   0.5713   0.0009   0.6475   0.9993   0.4000  
0.35       0.4013   0.5717   0.0030   0.7988   0.9989   0.4004  
0.40       0.4043   0.5718   0.0111   0.7399   0.9942   0.4013   <--
0.45       0.4112   0.5645   0.0493   0.6161   0.9539   0.4008  
0.50       0.5890   0.0001   0.9816   0.5956   0.0001   0.0025  
0.55       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.60       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.65       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.4043, F1=0.5718, Normal Recall=0.0111, Normal Precision=0.7399, Attack Recall=0.9942, Attack Precision=0.4013

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5002   0.6667   0.0004   0.9231   1.0000   0.5001  
0.20       0.5003   0.6668   0.0005   0.9412   1.0000   0.5001  
0.25       0.5003   0.6667   0.0008   0.7705   0.9998   0.5001  
0.30       0.5001   0.6665   0.0009   0.5376   0.9993   0.5000  
0.35       0.5008   0.6668   0.0028   0.7137   0.9989   0.5004   <--
0.40       0.5027   0.6665   0.0111   0.6562   0.9942   0.5013  
0.45       0.5013   0.6567   0.0488   0.5142   0.9539   0.5007  
0.50       0.4910   0.0001   0.9820   0.4955   0.0001   0.0038  
0.55       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.60       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.65       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.5008, F1=0.6668, Normal Recall=0.0028, Normal Precision=0.7137, Attack Recall=0.9989, Attack Precision=0.5004

```


## Threshold Tuning (saved_model_pruned_qat)

Model: `models/tflite/saved_model_pruned_qat.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9464   0.7775   0.9474   0.9927   0.9371   0.6643  
0.20       0.9614   0.8272   0.9657   0.9912   0.9229   0.7495  
0.25       0.9637   0.8353   0.9686   0.9909   0.9197   0.7651  
0.30       0.9662   0.8446   0.9715   0.9907   0.9184   0.7818  
0.35       0.9671   0.8479   0.9727   0.9906   0.9169   0.7886  
0.40       0.9676   0.8496   0.9735   0.9904   0.9147   0.7932   <--
0.45       0.9669   0.8443   0.9747   0.9884   0.8969   0.7976  
0.50       0.9667   0.8432   0.9748   0.9881   0.8940   0.7978  
0.55       0.9674   0.8452   0.9760   0.9876   0.8899   0.8048  
0.60       0.9673   0.8440   0.9765   0.9871   0.8847   0.8068  
0.65       0.9669   0.8413   0.9770   0.9862   0.8766   0.8087  
0.70       0.9591   0.7893   0.9805   0.9742   0.7661   0.8139  
0.75       0.9675   0.8137   0.9959   0.9688   0.7110   0.9512  
0.80       0.9577   0.7351   0.9990   0.9560   0.5862   0.9854  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9676, F1=0.8496, Normal Recall=0.9735, Normal Precision=0.9904, Attack Recall=0.9147, Attack Precision=0.7932

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9452   0.8723   0.9475   0.9834   0.9360   0.8167  
0.20       0.9570   0.8955   0.9658   0.9801   0.9217   0.8706  
0.25       0.9587   0.8989   0.9686   0.9795   0.9188   0.8799  
0.30       0.9607   0.9033   0.9715   0.9792   0.9174   0.8896  
0.35       0.9614   0.9046   0.9728   0.9788   0.9159   0.8937  
0.40       0.9616   0.9050   0.9735   0.9784   0.9140   0.8962   <--
0.45       0.9591   0.8976   0.9747   0.9742   0.8966   0.8987  
0.50       0.9586   0.8962   0.9749   0.9734   0.8935   0.8989  
0.55       0.9588   0.8963   0.9760   0.9726   0.8900   0.9028  
0.60       0.9582   0.8944   0.9765   0.9714   0.8851   0.9040  
0.65       0.9569   0.8907   0.9770   0.9694   0.8768   0.9049  
0.70       0.9376   0.8306   0.9806   0.9435   0.7653   0.9081  
0.75       0.9388   0.8226   0.9960   0.9321   0.7098   0.9779  
0.80       0.9161   0.7360   0.9991   0.9058   0.5845   0.9936  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9616, F1=0.9050, Normal Recall=0.9735, Normal Precision=0.9784, Attack Recall=0.9140, Attack Precision=0.8962

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9440   0.9093   0.9474   0.9719   0.9360   0.8842  
0.20       0.9525   0.9209   0.9657   0.9664   0.9217   0.9201  
0.25       0.9536   0.9223   0.9685   0.9653   0.9188   0.9259  
0.30       0.9552   0.9247   0.9713   0.9649   0.9174   0.9321  
0.35       0.9556   0.9252   0.9726   0.9643   0.9159   0.9347   <--
0.40       0.9555   0.9250   0.9734   0.9635   0.9140   0.9363  
0.45       0.9512   0.9168   0.9746   0.9565   0.8965   0.9379  
0.50       0.9503   0.9152   0.9747   0.9553   0.8935   0.9380  
0.55       0.9501   0.9145   0.9758   0.9539   0.8900   0.9404  
0.60       0.9489   0.9122   0.9763   0.9520   0.8851   0.9411  
0.65       0.9467   0.9080   0.9766   0.9487   0.8768   0.9415  
0.70       0.9159   0.8453   0.9805   0.9070   0.7653   0.9439  
0.75       0.9102   0.8259   0.9961   0.8890   0.7098   0.9874  
0.80       0.8747   0.7368   0.9991   0.8487   0.5845   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9556, F1=0.9252, Normal Recall=0.9726, Normal Precision=0.9643, Attack Recall=0.9159, Attack Precision=0.9347

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9430   0.9293   0.9477   0.9569   0.9360   0.9226  
0.20       0.9481   0.9343   0.9657   0.9487   0.9217   0.9471  
0.25       0.9487   0.9347   0.9686   0.9471   0.9188   0.9512  
0.30       0.9498   0.9360   0.9713   0.9464   0.9174   0.9552  
0.35       0.9500   0.9361   0.9727   0.9455   0.9159   0.9572   <--
0.40       0.9497   0.9356   0.9735   0.9444   0.9140   0.9583  
0.45       0.9434   0.9269   0.9746   0.9339   0.8966   0.9593  
0.50       0.9423   0.9253   0.9748   0.9321   0.8935   0.9594  
0.55       0.9416   0.9242   0.9760   0.9301   0.8900   0.9611  
0.60       0.9399   0.9217   0.9764   0.9272   0.8851   0.9615  
0.65       0.9367   0.9173   0.9767   0.9225   0.8768   0.9616  
0.70       0.8946   0.8531   0.9807   0.8624   0.7653   0.9636  
0.75       0.8816   0.8275   0.9962   0.8374   0.7098   0.9920  
0.80       0.8332   0.7371   0.9990   0.7829   0.5845   0.9975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9500, F1=0.9361, Normal Recall=0.9727, Normal Precision=0.9455, Attack Recall=0.9159, Attack Precision=0.9572

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9418   0.9415   0.9477   0.9367   0.9360   0.9471  
0.20       0.9438   0.9426   0.9659   0.9251   0.9217   0.9643  
0.25       0.9439   0.9425   0.9690   0.9227   0.9188   0.9674  
0.30       0.9446   0.9430   0.9717   0.9217   0.9174   0.9701   <--
0.35       0.9445   0.9429   0.9732   0.9204   0.9159   0.9715  
0.40       0.9439   0.9422   0.9738   0.9188   0.9140   0.9721  
0.45       0.9357   0.9331   0.9749   0.9041   0.8966   0.9728  
0.50       0.9343   0.9315   0.9751   0.9016   0.8935   0.9729  
0.55       0.9331   0.9301   0.9763   0.8987   0.8900   0.9741  
0.60       0.9309   0.9276   0.9768   0.8947   0.8851   0.9744  
0.65       0.9269   0.9231   0.9770   0.8881   0.8768   0.9744  
0.70       0.8732   0.8579   0.9811   0.8070   0.7653   0.9759  
0.75       0.8530   0.8284   0.9962   0.7744   0.7098   0.9947  
0.80       0.7918   0.7373   0.9990   0.7062   0.5845   0.9984  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9446, F1=0.9430, Normal Recall=0.9717, Normal Precision=0.9217, Attack Recall=0.9174, Attack Precision=0.9701

```

