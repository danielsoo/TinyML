# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-22 10:43:04 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8072 | 0.8266 | 0.8451 | 0.8647 | 0.8841 | 0.9021 | 0.9220 | 0.9410 | 0.9591 | 0.9785 | 0.9979 |
| QAT+Prune only | 0.1701 | 0.2531 | 0.3358 | 0.4187 | 0.5025 | 0.5840 | 0.6681 | 0.7509 | 0.8342 | 0.9167 | 0.9999 |
| QAT+PTQ | 0.1696 | 0.2526 | 0.3354 | 0.4183 | 0.5022 | 0.5837 | 0.6678 | 0.7506 | 0.8341 | 0.9168 | 0.9999 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.1696 | 0.2526 | 0.3354 | 0.4183 | 0.5022 | 0.5837 | 0.6678 | 0.7506 | 0.8341 | 0.9168 | 0.9999 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5351 | 0.7204 | 0.8157 | 0.8732 | 0.9106 | 0.9389 | 0.9595 | 0.9750 | 0.9882 | 0.9990 |
| QAT+Prune only | 0.0000 | 0.2112 | 0.3758 | 0.5079 | 0.6165 | 0.7062 | 0.7833 | 0.8489 | 0.9061 | 0.9558 | 0.9999 |
| QAT+PTQ | 0.0000 | 0.2111 | 0.3757 | 0.5077 | 0.6164 | 0.7060 | 0.7832 | 0.8488 | 0.9061 | 0.9558 | 0.9999 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2111 | 0.3757 | 0.5077 | 0.6164 | 0.7060 | 0.7832 | 0.8488 | 0.9061 | 0.9558 | 0.9999 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8072 | 0.8075 | 0.8068 | 0.8076 | 0.8082 | 0.8062 | 0.8082 | 0.8081 | 0.8040 | 0.8037 | 0.0000 |
| QAT+Prune only | 0.1701 | 0.1701 | 0.1698 | 0.1696 | 0.1709 | 0.1682 | 0.1705 | 0.1698 | 0.1716 | 0.1687 | 0.0000 |
| QAT+PTQ | 0.1696 | 0.1696 | 0.1693 | 0.1690 | 0.1704 | 0.1675 | 0.1698 | 0.1691 | 0.1713 | 0.1688 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.1696 | 0.1696 | 0.1693 | 0.1690 | 0.1704 | 0.1675 | 0.1698 | 0.1691 | 0.1713 | 0.1688 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8072 | 0.0000 | 0.0000 | 0.0000 | 0.8072 | 1.0000 |
| 90 | 10 | 299,940 | 0.8266 | 0.3655 | 0.9980 | 0.5351 | 0.8075 | 0.9997 |
| 80 | 20 | 291,350 | 0.8451 | 0.5636 | 0.9979 | 0.7204 | 0.8068 | 0.9994 |
| 70 | 30 | 194,230 | 0.8647 | 0.6897 | 0.9979 | 0.8157 | 0.8076 | 0.9989 |
| 60 | 40 | 145,675 | 0.8841 | 0.7762 | 0.9979 | 0.8732 | 0.8082 | 0.9983 |
| 50 | 50 | 116,540 | 0.9021 | 0.8374 | 0.9979 | 0.9106 | 0.8062 | 0.9974 |
| 40 | 60 | 97,115 | 0.9220 | 0.8864 | 0.9979 | 0.9389 | 0.8082 | 0.9962 |
| 30 | 70 | 83,240 | 0.9410 | 0.9239 | 0.9979 | 0.9595 | 0.8081 | 0.9940 |
| 20 | 80 | 72,835 | 0.9591 | 0.9532 | 0.9979 | 0.9750 | 0.8040 | 0.9898 |
| 10 | 90 | 64,740 | 0.9785 | 0.9786 | 0.9979 | 0.9882 | 0.8037 | 0.9773 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9990 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1701 | 0.0000 | 0.0000 | 0.0000 | 0.1701 | 1.0000 |
| 90 | 10 | 299,940 | 0.2531 | 0.1181 | 0.9999 | 0.2112 | 0.1701 | 1.0000 |
| 80 | 20 | 291,350 | 0.3358 | 0.2314 | 0.9999 | 0.3758 | 0.1698 | 0.9998 |
| 70 | 30 | 194,230 | 0.4187 | 0.3404 | 0.9999 | 0.5079 | 0.1696 | 0.9997 |
| 60 | 40 | 145,675 | 0.5025 | 0.4457 | 0.9999 | 0.6165 | 0.1709 | 0.9995 |
| 50 | 50 | 116,540 | 0.5840 | 0.5459 | 0.9999 | 0.7062 | 0.1682 | 0.9992 |
| 40 | 60 | 97,115 | 0.6681 | 0.6439 | 0.9999 | 0.7833 | 0.1705 | 0.9988 |
| 30 | 70 | 83,240 | 0.7509 | 0.7376 | 0.9999 | 0.8489 | 0.1698 | 0.9981 |
| 20 | 80 | 72,835 | 0.8342 | 0.8284 | 0.9999 | 0.9061 | 0.1716 | 0.9968 |
| 10 | 90 | 64,740 | 0.9167 | 0.9154 | 0.9999 | 0.9558 | 0.1687 | 0.9927 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1696 | 0.0000 | 0.0000 | 0.0000 | 0.1696 | 1.0000 |
| 90 | 10 | 299,940 | 0.2526 | 0.1180 | 0.9999 | 0.2111 | 0.1696 | 1.0000 |
| 80 | 20 | 291,350 | 0.3354 | 0.2313 | 0.9999 | 0.3757 | 0.1693 | 0.9998 |
| 70 | 30 | 194,230 | 0.4183 | 0.3402 | 0.9999 | 0.5077 | 0.1690 | 0.9997 |
| 60 | 40 | 145,675 | 0.5022 | 0.4455 | 0.9999 | 0.6164 | 0.1704 | 0.9995 |
| 50 | 50 | 116,540 | 0.5837 | 0.5457 | 0.9999 | 0.7060 | 0.1675 | 0.9992 |
| 40 | 60 | 97,115 | 0.6678 | 0.6437 | 0.9999 | 0.7832 | 0.1698 | 0.9988 |
| 30 | 70 | 83,240 | 0.7506 | 0.7374 | 0.9999 | 0.8488 | 0.1691 | 0.9981 |
| 20 | 80 | 72,835 | 0.8341 | 0.8284 | 0.9999 | 0.9061 | 0.1713 | 0.9968 |
| 10 | 90 | 64,740 | 0.9168 | 0.9154 | 0.9999 | 0.9558 | 0.1688 | 0.9927 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,940 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.9000 |
| 80 | 20 | 291,350 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8000 |
| 70 | 30 | 194,230 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7000 |
| 60 | 40 | 145,675 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 |
| 50 | 50 | 116,540 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 40 | 60 | 97,115 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4000 |
| 30 | 70 | 83,240 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3000 |
| 20 | 80 | 72,835 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 10 | 90 | 64,740 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |
| 0 | 100 | 58,270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Compressed (PTQ)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1696 | 0.0000 | 0.0000 | 0.0000 | 0.1696 | 1.0000 |
| 90 | 10 | 299,940 | 0.2526 | 0.1180 | 0.9999 | 0.2111 | 0.1696 | 1.0000 |
| 80 | 20 | 291,350 | 0.3354 | 0.2313 | 0.9999 | 0.3757 | 0.1693 | 0.9998 |
| 70 | 30 | 194,230 | 0.4183 | 0.3402 | 0.9999 | 0.5077 | 0.1690 | 0.9997 |
| 60 | 40 | 145,675 | 0.5022 | 0.4455 | 0.9999 | 0.6164 | 0.1704 | 0.9995 |
| 50 | 50 | 116,540 | 0.5837 | 0.5457 | 0.9999 | 0.7060 | 0.1675 | 0.9992 |
| 40 | 60 | 97,115 | 0.6678 | 0.6437 | 0.9999 | 0.7832 | 0.1698 | 0.9988 |
| 30 | 70 | 83,240 | 0.7506 | 0.7374 | 0.9999 | 0.8488 | 0.1691 | 0.9981 |
| 20 | 80 | 72,835 | 0.8341 | 0.8284 | 0.9999 | 0.9061 | 0.1713 | 0.9968 |
| 10 | 90 | 64,740 | 0.9168 | 0.9154 | 0.9999 | 0.9558 | 0.1688 | 0.9927 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0000 |


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
0.15       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656   <--
0.20       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.25       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.30       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.35       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.40       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.45       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.50       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.55       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.60       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.65       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.70       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.75       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
0.80       0.8266   0.5351   0.8075   0.9997   0.9981   0.3656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8266, F1=0.5351, Normal Recall=0.8075, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.3656

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
0.15       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652   <--
0.20       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.25       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.30       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.35       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.40       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.45       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.50       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.55       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.60       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.65       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.70       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.75       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
0.80       0.8461   0.7217   0.8081   0.9994   0.9979   0.5652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8461, F1=0.7217, Normal Recall=0.8081, Normal Precision=0.9994, Attack Recall=0.9979, Attack Precision=0.5652

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
0.15       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898   <--
0.20       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.25       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.30       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.35       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.40       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.45       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.50       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.55       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.60       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.65       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.70       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.75       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
0.80       0.8647   0.8157   0.8077   0.9989   0.9979   0.6898  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8647, F1=0.8157, Normal Recall=0.8077, Normal Precision=0.9989, Attack Recall=0.9979, Attack Precision=0.6898

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
0.15       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753   <--
0.20       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.25       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.30       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.35       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.40       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.45       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.50       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.55       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.60       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.65       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.70       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.75       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
0.80       0.8835   0.8726   0.8072   0.9983   0.9979   0.7753  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8835, F1=0.8726, Normal Recall=0.8072, Normal Precision=0.9983, Attack Recall=0.9979, Attack Precision=0.7753

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
0.15       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367   <--
0.20       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.25       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.30       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.35       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.40       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.45       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.50       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.55       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.60       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.65       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.70       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.75       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
0.80       0.9016   0.9102   0.8052   0.9974   0.9979   0.8367  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9016, F1=0.9102, Normal Recall=0.8052, Normal Precision=0.9974, Attack Recall=0.9979, Attack Precision=0.8367

```


## Threshold Tuning (QAT+Prune only)

Model: `models/tflite/saved_model_qat_pruned_float32.tflite`

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
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181   <--
0.20       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.25       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.30       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.35       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.40       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.45       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.50       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.55       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.60       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.65       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.70       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.75       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
0.80       0.2531   0.2112   0.1701   0.9999   0.9999   0.1181  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2531, F1=0.2112, Normal Recall=0.1701, Normal Precision=0.9999, Attack Recall=0.9999, Attack Precision=0.1181

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
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315   <--
0.20       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.25       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.30       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.35       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.40       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.45       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.50       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.55       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.60       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.65       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.70       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.75       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
0.80       0.3362   0.3760   0.1703   0.9998   0.9999   0.2315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3362, F1=0.3760, Normal Recall=0.1703, Normal Precision=0.9998, Attack Recall=0.9999, Attack Precision=0.2315

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
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407   <--
0.20       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.25       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.30       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.35       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.40       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.45       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.50       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.55       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.60       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.65       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.70       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.75       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
0.80       0.4194   0.5082   0.1706   0.9997   0.9999   0.3407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4194, F1=0.5082, Normal Recall=0.1706, Normal Precision=0.9997, Attack Recall=0.9999, Attack Precision=0.3407

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
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454   <--
0.20       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.25       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.30       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.35       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.40       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.45       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.50       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.55       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.60       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.65       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.70       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.75       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
0.80       0.5020   0.6163   0.1700   0.9995   0.9999   0.4454  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5020, F1=0.6163, Normal Recall=0.1700, Normal Precision=0.9995, Attack Recall=0.9999, Attack Precision=0.4454

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
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459   <--
0.20       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.25       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.30       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.35       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.40       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.45       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.50       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.55       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.60       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.65       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.70       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.75       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
0.80       0.5841   0.7063   0.1684   0.9992   0.9999   0.5459  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5841, F1=0.7063, Normal Recall=0.1684, Normal Precision=0.9992, Attack Recall=0.9999, Attack Precision=0.5459

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
0.15       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180   <--
0.20       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.25       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.30       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.35       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.40       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.45       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.50       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.55       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.60       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.65       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.70       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.75       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.80       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2526, F1=0.2111, Normal Recall=0.1696, Normal Precision=0.9999, Attack Recall=0.9999, Attack Precision=0.1180

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
0.15       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314   <--
0.20       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.25       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.30       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.35       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.40       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.45       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.50       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.55       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.60       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.65       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.70       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.75       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.80       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3357, F1=0.3758, Normal Recall=0.1697, Normal Precision=0.9998, Attack Recall=0.9999, Attack Precision=0.2314

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
0.15       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405   <--
0.20       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.25       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.30       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.35       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.40       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.45       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.50       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.55       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.60       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.65       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.70       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.75       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.80       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4190, F1=0.5080, Normal Recall=0.1701, Normal Precision=0.9997, Attack Recall=0.9999, Attack Precision=0.3405

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
0.15       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453   <--
0.20       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.25       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.30       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.35       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.40       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.45       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.50       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.55       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.60       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.65       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.70       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.75       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.80       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5017, F1=0.6161, Normal Recall=0.1695, Normal Precision=0.9995, Attack Recall=0.9999, Attack Precision=0.4453

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
0.15       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458   <--
0.20       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.25       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.30       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.35       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.40       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.45       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.50       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.55       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.60       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.65       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.70       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.75       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.80       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5839, F1=0.7061, Normal Recall=0.1679, Normal Precision=0.9992, Attack Recall=0.9999, Attack Precision=0.5458

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
0.15       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000   <--
0.20       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.25       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.30       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.35       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.40       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.45       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.50       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.55       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.60       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.65       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.70       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.9000, Attack Recall=0.0000, Attack Precision=0.0000

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
0.15       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000   <--
0.20       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.25       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.30       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.35       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.40       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.45       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.50       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.55       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.60       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.65       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.70       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.8000, Attack Recall=0.0000, Attack Precision=0.0000

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
0.15       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000   <--
0.20       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.25       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.30       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.35       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.40       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.45       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.50       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.55       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.60       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.65       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.7000, Attack Recall=0.0000, Attack Precision=0.0000

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
0.15       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000   <--
0.20       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.25       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.30       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.35       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.40       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.45       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.50       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.55       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.60       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.65       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.6000, Attack Recall=0.0000, Attack Precision=0.0000

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
0.15       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000   <--
0.20       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.25       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.30       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.35       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.40       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.45       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.50       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.55       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.60       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.65       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.5000, Attack Recall=0.0000, Attack Precision=0.0000

```


## Threshold Tuning (Compressed (PTQ))

Model: `models/tflite/saved_model_pruned_quantized.tflite`

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
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180   <--
0.20       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.25       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.30       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.35       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.40       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.45       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.50       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.55       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.60       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.65       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.70       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.75       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
0.80       0.2526   0.2111   0.1696   0.9999   0.9999   0.1180  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2526, F1=0.2111, Normal Recall=0.1696, Normal Precision=0.9999, Attack Recall=0.9999, Attack Precision=0.1180

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
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314   <--
0.20       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.25       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.30       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.35       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.40       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.45       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.50       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.55       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.60       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.65       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.70       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.75       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
0.80       0.3357   0.3758   0.1697   0.9998   0.9999   0.2314  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3357, F1=0.3758, Normal Recall=0.1697, Normal Precision=0.9998, Attack Recall=0.9999, Attack Precision=0.2314

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
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405   <--
0.20       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.25       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.30       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.35       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.40       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.45       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.50       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.55       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.60       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.65       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.70       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.75       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
0.80       0.4190   0.5080   0.1701   0.9997   0.9999   0.3405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4190, F1=0.5080, Normal Recall=0.1701, Normal Precision=0.9997, Attack Recall=0.9999, Attack Precision=0.3405

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
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453   <--
0.20       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.25       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.30       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.35       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.40       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.45       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.50       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.55       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.60       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.65       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.70       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.75       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
0.80       0.5017   0.6161   0.1695   0.9995   0.9999   0.4453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5017, F1=0.6161, Normal Recall=0.1695, Normal Precision=0.9995, Attack Recall=0.9999, Attack Precision=0.4453

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
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458   <--
0.20       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.25       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.30       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.35       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.40       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.45       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.50       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.55       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.60       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.65       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.70       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.75       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
0.80       0.5839   0.7061   0.1679   0.9992   0.9999   0.5458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5839, F1=0.7061, Normal Recall=0.1679, Normal Precision=0.9992, Attack Recall=0.9999, Attack Precision=0.5458

```

