# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-13 16:20:08 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4377 | 0.4538 | 0.4702 | 0.4873 | 0.5038 | 0.5200 | 0.5361 | 0.5533 | 0.5690 | 0.5867 | 0.6026 |
| QAT+Prune only | 0.7390 | 0.7642 | 0.7894 | 0.8160 | 0.8426 | 0.8667 | 0.8940 | 0.9193 | 0.9461 | 0.9714 | 0.9980 |
| QAT+PTQ | 0.7379 | 0.7631 | 0.7885 | 0.8150 | 0.8419 | 0.8661 | 0.8935 | 0.9190 | 0.9456 | 0.9713 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7379 | 0.7631 | 0.7885 | 0.8150 | 0.8419 | 0.8661 | 0.8935 | 0.9190 | 0.9456 | 0.9713 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1808 | 0.3127 | 0.4136 | 0.4928 | 0.5566 | 0.6092 | 0.6538 | 0.6911 | 0.7241 | 0.7520 |
| QAT+Prune only | 0.0000 | 0.4584 | 0.6547 | 0.7649 | 0.8353 | 0.8822 | 0.9187 | 0.9454 | 0.9673 | 0.9843 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4573 | 0.6537 | 0.7640 | 0.8347 | 0.8817 | 0.9183 | 0.9452 | 0.9671 | 0.9843 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4573 | 0.6537 | 0.7640 | 0.8347 | 0.8817 | 0.9183 | 0.9452 | 0.9671 | 0.9843 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4377 | 0.4373 | 0.4370 | 0.4379 | 0.4380 | 0.4374 | 0.4362 | 0.4384 | 0.4348 | 0.4436 | 0.0000 |
| QAT+Prune only | 0.7390 | 0.7382 | 0.7373 | 0.7379 | 0.7389 | 0.7355 | 0.7379 | 0.7357 | 0.7382 | 0.7319 | 0.0000 |
| QAT+PTQ | 0.7379 | 0.7370 | 0.7361 | 0.7366 | 0.7377 | 0.7343 | 0.7367 | 0.7347 | 0.7361 | 0.7309 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7379 | 0.7370 | 0.7361 | 0.7366 | 0.7377 | 0.7343 | 0.7367 | 0.7347 | 0.7361 | 0.7309 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4377 | 0.0000 | 0.0000 | 0.0000 | 0.4377 | 1.0000 |
| 90 | 10 | 299,940 | 0.4538 | 0.1063 | 0.6027 | 0.1808 | 0.4373 | 0.9083 |
| 80 | 20 | 291,350 | 0.4702 | 0.2111 | 0.6026 | 0.3127 | 0.4370 | 0.8148 |
| 70 | 30 | 194,230 | 0.4873 | 0.3148 | 0.6026 | 0.4136 | 0.4379 | 0.7200 |
| 60 | 40 | 145,675 | 0.5038 | 0.4168 | 0.6026 | 0.4928 | 0.4380 | 0.6231 |
| 50 | 50 | 116,540 | 0.5200 | 0.5172 | 0.6026 | 0.5566 | 0.4374 | 0.5240 |
| 40 | 60 | 97,115 | 0.5361 | 0.6159 | 0.6026 | 0.6092 | 0.4362 | 0.4226 |
| 30 | 70 | 83,240 | 0.5533 | 0.7146 | 0.6026 | 0.6538 | 0.4384 | 0.3210 |
| 20 | 80 | 72,835 | 0.5690 | 0.8101 | 0.6026 | 0.6911 | 0.4348 | 0.2148 |
| 10 | 90 | 64,740 | 0.5867 | 0.9070 | 0.6026 | 0.7241 | 0.4436 | 0.1103 |
| 0 | 100 | 58,270 | 0.6026 | 1.0000 | 0.6026 | 0.7520 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7390 | 0.0000 | 0.0000 | 0.0000 | 0.7390 | 1.0000 |
| 90 | 10 | 299,940 | 0.7642 | 0.2975 | 0.9980 | 0.4584 | 0.7382 | 0.9997 |
| 80 | 20 | 291,350 | 0.7894 | 0.4871 | 0.9980 | 0.6547 | 0.7373 | 0.9993 |
| 70 | 30 | 194,230 | 0.8160 | 0.6201 | 0.9980 | 0.7649 | 0.7379 | 0.9989 |
| 60 | 40 | 145,675 | 0.8426 | 0.7182 | 0.9980 | 0.8353 | 0.7389 | 0.9982 |
| 50 | 50 | 116,540 | 0.8667 | 0.7905 | 0.9980 | 0.8822 | 0.7355 | 0.9973 |
| 40 | 60 | 97,115 | 0.8940 | 0.8510 | 0.9980 | 0.9187 | 0.7379 | 0.9960 |
| 30 | 70 | 83,240 | 0.9193 | 0.8981 | 0.9980 | 0.9454 | 0.7357 | 0.9938 |
| 20 | 80 | 72,835 | 0.9461 | 0.9385 | 0.9980 | 0.9673 | 0.7382 | 0.9894 |
| 10 | 90 | 64,740 | 0.9714 | 0.9710 | 0.9980 | 0.9843 | 0.7319 | 0.9763 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7379 | 0.0000 | 0.0000 | 0.0000 | 0.7379 | 1.0000 |
| 90 | 10 | 299,940 | 0.7631 | 0.2966 | 0.9980 | 0.4573 | 0.7370 | 0.9997 |
| 80 | 20 | 291,350 | 0.7885 | 0.4860 | 0.9980 | 0.6537 | 0.7361 | 0.9993 |
| 70 | 30 | 194,230 | 0.8150 | 0.6189 | 0.9980 | 0.7640 | 0.7366 | 0.9989 |
| 60 | 40 | 145,675 | 0.8419 | 0.7173 | 0.9980 | 0.8347 | 0.7377 | 0.9982 |
| 50 | 50 | 116,540 | 0.8661 | 0.7897 | 0.9980 | 0.8817 | 0.7343 | 0.9973 |
| 40 | 60 | 97,115 | 0.8935 | 0.8504 | 0.9980 | 0.9183 | 0.7367 | 0.9960 |
| 30 | 70 | 83,240 | 0.9190 | 0.8977 | 0.9980 | 0.9452 | 0.7347 | 0.9938 |
| 20 | 80 | 72,835 | 0.9456 | 0.9380 | 0.9980 | 0.9671 | 0.7361 | 0.9894 |
| 10 | 90 | 64,740 | 0.9713 | 0.9709 | 0.9980 | 0.9843 | 0.7309 | 0.9763 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7379 | 0.0000 | 0.0000 | 0.0000 | 0.7379 | 1.0000 |
| 90 | 10 | 299,940 | 0.7631 | 0.2966 | 0.9980 | 0.4573 | 0.7370 | 0.9997 |
| 80 | 20 | 291,350 | 0.7885 | 0.4860 | 0.9980 | 0.6537 | 0.7361 | 0.9993 |
| 70 | 30 | 194,230 | 0.8150 | 0.6189 | 0.9980 | 0.7640 | 0.7366 | 0.9989 |
| 60 | 40 | 145,675 | 0.8419 | 0.7173 | 0.9980 | 0.8347 | 0.7377 | 0.9982 |
| 50 | 50 | 116,540 | 0.8661 | 0.7897 | 0.9980 | 0.8817 | 0.7343 | 0.9973 |
| 40 | 60 | 97,115 | 0.8935 | 0.8504 | 0.9980 | 0.9183 | 0.7367 | 0.9960 |
| 30 | 70 | 83,240 | 0.9190 | 0.8977 | 0.9980 | 0.9452 | 0.7347 | 0.9938 |
| 20 | 80 | 72,835 | 0.9456 | 0.9380 | 0.9980 | 0.9671 | 0.7361 | 0.9894 |
| 10 | 90 | 64,740 | 0.9713 | 0.9709 | 0.9980 | 0.9843 | 0.7309 | 0.9763 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060   <--
0.20       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.25       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.30       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.35       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.40       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.45       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.50       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.55       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.60       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.65       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.70       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.75       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
0.80       0.4536   0.1802   0.4373   0.9078   0.6005   0.1060  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4536, F1=0.1802, Normal Recall=0.4373, Normal Precision=0.9078, Attack Recall=0.6005, Attack Precision=0.1060

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
0.15       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109   <--
0.20       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.25       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.30       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.35       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.40       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.45       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.50       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.55       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.60       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.65       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.70       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.75       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
0.80       0.4695   0.3124   0.4362   0.8145   0.6026   0.2109  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4695, F1=0.3124, Normal Recall=0.4362, Normal Precision=0.8145, Attack Recall=0.6026, Attack Precision=0.2109

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
0.15       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146   <--
0.20       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.25       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.30       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.35       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.40       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.45       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.50       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.55       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.60       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.65       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.70       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.75       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
0.80       0.4869   0.4134   0.4374   0.7197   0.6026   0.3146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4869, F1=0.4134, Normal Recall=0.4374, Normal Precision=0.7197, Attack Recall=0.6026, Attack Precision=0.3146

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
0.15       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168   <--
0.20       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.25       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.30       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.35       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.40       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.45       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.50       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.55       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.60       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.65       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.70       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.75       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
0.80       0.5038   0.4928   0.4379   0.6231   0.6026   0.4168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5038, F1=0.4928, Normal Recall=0.4379, Normal Precision=0.6231, Attack Recall=0.6026, Attack Precision=0.4168

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
0.15       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179   <--
0.20       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.25       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.30       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.35       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.40       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.45       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.50       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.55       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.60       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.65       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.70       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.75       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
0.80       0.5208   0.5570   0.4390   0.5249   0.6026   0.5179  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5208, F1=0.5570, Normal Recall=0.4390, Normal Precision=0.5249, Attack Recall=0.6026, Attack Precision=0.5179

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
0.15       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975   <--
0.20       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.25       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.30       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.35       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.40       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.45       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.50       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.55       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.60       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.65       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.70       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.75       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
0.80       0.7642   0.4584   0.7382   0.9997   0.9981   0.2975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7642, F1=0.4584, Normal Recall=0.7382, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2975

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
0.15       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887   <--
0.20       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.25       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.30       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.35       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.40       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.45       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.50       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.55       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.60       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.65       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.70       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.75       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
0.80       0.7908   0.6561   0.7389   0.9993   0.9980   0.4887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7908, F1=0.6561, Normal Recall=0.7389, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4887

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
0.15       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215   <--
0.20       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.25       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.30       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.35       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.40       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.45       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.50       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.55       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.60       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.65       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.70       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.75       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
0.80       0.8171   0.7660   0.7395   0.9989   0.9980   0.6215  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8171, F1=0.7660, Normal Recall=0.7395, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6215

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
0.15       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180   <--
0.20       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.25       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.30       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.35       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.40       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.45       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.50       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.55       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.60       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.65       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.70       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.75       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
0.80       0.8424   0.8352   0.7387   0.9982   0.9980   0.7180  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8424, F1=0.8352, Normal Recall=0.7387, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7180

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
0.15       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917   <--
0.20       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.25       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.30       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.35       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.40       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.45       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.50       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.55       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.60       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.65       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.70       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.75       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
0.80       0.8677   0.8829   0.7374   0.9973   0.9980   0.7917  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8677, F1=0.8829, Normal Recall=0.7374, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7917

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
0.15       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966   <--
0.20       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.25       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.30       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.35       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.40       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.45       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.50       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.55       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.60       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.65       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.70       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.75       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.80       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7631, F1=0.4573, Normal Recall=0.7370, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2966

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
0.15       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876   <--
0.20       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.25       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.30       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.35       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.40       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.45       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.50       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.55       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.60       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.65       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.70       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.75       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.80       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7898, F1=0.6551, Normal Recall=0.7378, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4876

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
0.15       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204   <--
0.20       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.25       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.30       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.35       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.40       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.45       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.50       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.55       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.60       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.65       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.70       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.75       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.80       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8162, F1=0.7652, Normal Recall=0.7383, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6204

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
0.15       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172   <--
0.20       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.25       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.30       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.35       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.40       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.45       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.50       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.55       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.60       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.65       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.70       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.75       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.80       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8418, F1=0.8346, Normal Recall=0.7376, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7172

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
0.15       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911   <--
0.20       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.25       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.30       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.35       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.40       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.45       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.50       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.55       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.60       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.65       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.70       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.75       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.80       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8672, F1=0.8826, Normal Recall=0.7364, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7911

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
0.15       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966   <--
0.20       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.25       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.30       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.35       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.40       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.45       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.50       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.55       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.60       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.65       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.70       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.75       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
0.80       0.7631   0.4573   0.7370   0.9997   0.9981   0.2966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7631, F1=0.4573, Normal Recall=0.7370, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2966

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
0.15       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876   <--
0.20       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.25       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.30       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.35       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.40       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.45       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.50       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.55       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.60       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.65       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.70       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.75       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
0.80       0.7898   0.6551   0.7378   0.9993   0.9980   0.4876  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7898, F1=0.6551, Normal Recall=0.7378, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4876

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
0.15       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204   <--
0.20       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.25       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.30       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.35       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.40       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.45       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.50       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.55       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.60       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.65       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.70       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.75       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
0.80       0.8162   0.7652   0.7383   0.9989   0.9980   0.6204  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8162, F1=0.7652, Normal Recall=0.7383, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6204

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
0.15       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172   <--
0.20       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.25       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.30       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.35       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.40       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.45       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.50       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.55       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.60       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.65       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.70       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.75       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
0.80       0.8418   0.8346   0.7376   0.9982   0.9980   0.7172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8418, F1=0.8346, Normal Recall=0.7376, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7172

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
0.15       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911   <--
0.20       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.25       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.30       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.35       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.40       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.45       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.50       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.55       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.60       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.65       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.70       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.75       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
0.80       0.8672   0.8826   0.7364   0.9973   0.9980   0.7911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8672, F1=0.8826, Normal Recall=0.7364, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7911

```

