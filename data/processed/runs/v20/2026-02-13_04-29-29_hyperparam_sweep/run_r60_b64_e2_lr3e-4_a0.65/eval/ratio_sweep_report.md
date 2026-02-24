# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-13 17:59:17 |

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
| Original (TFLite) | 0.9328 | 0.8743 | 0.8163 | 0.7582 | 0.7001 | 0.6423 | 0.5840 | 0.5261 | 0.4678 | 0.4100 | 0.3515 |
| QAT+Prune only | 0.9025 | 0.8956 | 0.8890 | 0.8830 | 0.8768 | 0.8695 | 0.8642 | 0.8575 | 0.8511 | 0.8451 | 0.8392 |
| QAT+PTQ | 0.9030 | 0.8962 | 0.8896 | 0.8837 | 0.8772 | 0.8700 | 0.8645 | 0.8577 | 0.8513 | 0.8453 | 0.8394 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9030 | 0.8962 | 0.8896 | 0.8837 | 0.8772 | 0.8700 | 0.8645 | 0.8577 | 0.8513 | 0.8453 | 0.8394 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3578 | 0.4336 | 0.4658 | 0.4839 | 0.4956 | 0.5035 | 0.5094 | 0.5137 | 0.5175 | 0.5201 |
| QAT+Prune only | 0.0000 | 0.6168 | 0.7514 | 0.8115 | 0.8449 | 0.8654 | 0.8812 | 0.8918 | 0.9002 | 0.9070 | 0.9126 |
| QAT+PTQ | 0.0000 | 0.6183 | 0.7525 | 0.8124 | 0.8454 | 0.8659 | 0.8814 | 0.8920 | 0.9003 | 0.9071 | 0.9127 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6183 | 0.7525 | 0.8124 | 0.8454 | 0.8659 | 0.8814 | 0.8920 | 0.9003 | 0.9071 | 0.9127 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9328 | 0.9325 | 0.9325 | 0.9325 | 0.9326 | 0.9330 | 0.9329 | 0.9337 | 0.9331 | 0.9370 | 0.0000 |
| QAT+Prune only | 0.9025 | 0.9018 | 0.9014 | 0.9018 | 0.9018 | 0.8998 | 0.9017 | 0.9000 | 0.8986 | 0.8984 | 0.0000 |
| QAT+PTQ | 0.9030 | 0.9025 | 0.9021 | 0.9027 | 0.9024 | 0.9005 | 0.9021 | 0.9005 | 0.8992 | 0.8985 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9030 | 0.9025 | 0.9021 | 0.9027 | 0.9024 | 0.9005 | 0.9021 | 0.9005 | 0.8992 | 0.8985 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9328 | 0.0000 | 0.0000 | 0.0000 | 0.9328 | 1.0000 |
| 90 | 10 | 299,940 | 0.8743 | 0.3657 | 0.3502 | 0.3578 | 0.9325 | 0.9281 |
| 80 | 20 | 291,350 | 0.8163 | 0.5657 | 0.3515 | 0.4336 | 0.9325 | 0.8519 |
| 70 | 30 | 194,230 | 0.7582 | 0.6906 | 0.3515 | 0.4658 | 0.9325 | 0.7704 |
| 60 | 40 | 145,675 | 0.7001 | 0.7766 | 0.3515 | 0.4839 | 0.9326 | 0.6832 |
| 50 | 50 | 116,540 | 0.6423 | 0.8400 | 0.3515 | 0.4956 | 0.9330 | 0.5899 |
| 40 | 60 | 97,115 | 0.5840 | 0.8872 | 0.3515 | 0.5035 | 0.9329 | 0.4895 |
| 30 | 70 | 83,240 | 0.5261 | 0.9252 | 0.3515 | 0.5094 | 0.9337 | 0.3816 |
| 20 | 80 | 72,835 | 0.4678 | 0.9546 | 0.3514 | 0.5137 | 0.9331 | 0.2645 |
| 10 | 90 | 64,740 | 0.4100 | 0.9805 | 0.3515 | 0.5175 | 0.9370 | 0.1383 |
| 0 | 100 | 58,270 | 0.3515 | 1.0000 | 0.3515 | 0.5201 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9025 | 0.0000 | 0.0000 | 0.0000 | 0.9025 | 1.0000 |
| 90 | 10 | 299,940 | 0.8956 | 0.4873 | 0.8401 | 0.6168 | 0.9018 | 0.9807 |
| 80 | 20 | 291,350 | 0.8890 | 0.6803 | 0.8392 | 0.7514 | 0.9014 | 0.9573 |
| 70 | 30 | 194,230 | 0.8830 | 0.7856 | 0.8392 | 0.8115 | 0.9018 | 0.9290 |
| 60 | 40 | 145,675 | 0.8768 | 0.8506 | 0.8392 | 0.8449 | 0.9018 | 0.8938 |
| 50 | 50 | 116,540 | 0.8695 | 0.8933 | 0.8392 | 0.8654 | 0.8998 | 0.8484 |
| 40 | 60 | 97,115 | 0.8642 | 0.9276 | 0.8392 | 0.8812 | 0.9017 | 0.7890 |
| 30 | 70 | 83,240 | 0.8575 | 0.9514 | 0.8392 | 0.8918 | 0.9000 | 0.7058 |
| 20 | 80 | 72,835 | 0.8511 | 0.9707 | 0.8392 | 0.9002 | 0.8986 | 0.5829 |
| 10 | 90 | 64,740 | 0.8451 | 0.9867 | 0.8392 | 0.9070 | 0.8984 | 0.3830 |
| 0 | 100 | 58,270 | 0.8392 | 1.0000 | 0.8392 | 0.9126 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9030 | 0.0000 | 0.0000 | 0.0000 | 0.9030 | 1.0000 |
| 90 | 10 | 299,940 | 0.8962 | 0.4891 | 0.8403 | 0.6183 | 0.9025 | 0.9807 |
| 80 | 20 | 291,350 | 0.8896 | 0.6819 | 0.8394 | 0.7525 | 0.9021 | 0.9574 |
| 70 | 30 | 194,230 | 0.8837 | 0.7871 | 0.8394 | 0.8124 | 0.9027 | 0.9291 |
| 60 | 40 | 145,675 | 0.8772 | 0.8515 | 0.8394 | 0.8454 | 0.9024 | 0.8939 |
| 50 | 50 | 116,540 | 0.8700 | 0.8941 | 0.8394 | 0.8659 | 0.9005 | 0.8486 |
| 40 | 60 | 97,115 | 0.8645 | 0.9278 | 0.8394 | 0.8814 | 0.9021 | 0.7892 |
| 30 | 70 | 83,240 | 0.8577 | 0.9516 | 0.8394 | 0.8920 | 0.9005 | 0.7061 |
| 20 | 80 | 72,835 | 0.8513 | 0.9708 | 0.8394 | 0.9003 | 0.8992 | 0.5833 |
| 10 | 90 | 64,740 | 0.8453 | 0.9867 | 0.8394 | 0.9071 | 0.8985 | 0.3833 |
| 0 | 100 | 58,270 | 0.8394 | 1.0000 | 0.8394 | 0.9127 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9030 | 0.0000 | 0.0000 | 0.0000 | 0.9030 | 1.0000 |
| 90 | 10 | 299,940 | 0.8962 | 0.4891 | 0.8403 | 0.6183 | 0.9025 | 0.9807 |
| 80 | 20 | 291,350 | 0.8896 | 0.6819 | 0.8394 | 0.7525 | 0.9021 | 0.9574 |
| 70 | 30 | 194,230 | 0.8837 | 0.7871 | 0.8394 | 0.8124 | 0.9027 | 0.9291 |
| 60 | 40 | 145,675 | 0.8772 | 0.8515 | 0.8394 | 0.8454 | 0.9024 | 0.8939 |
| 50 | 50 | 116,540 | 0.8700 | 0.8941 | 0.8394 | 0.8659 | 0.9005 | 0.8486 |
| 40 | 60 | 97,115 | 0.8645 | 0.9278 | 0.8394 | 0.8814 | 0.9021 | 0.7892 |
| 30 | 70 | 83,240 | 0.8577 | 0.9516 | 0.8394 | 0.8920 | 0.9005 | 0.7061 |
| 20 | 80 | 72,835 | 0.8513 | 0.9708 | 0.8394 | 0.9003 | 0.8992 | 0.5833 |
| 10 | 90 | 64,740 | 0.8453 | 0.9867 | 0.8394 | 0.9071 | 0.8985 | 0.3833 |
| 0 | 100 | 58,270 | 0.8394 | 1.0000 | 0.8394 | 0.9127 | 0.0000 | 0.0000 |


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
0.15       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644   <--
0.20       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.25       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.30       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.35       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.40       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.45       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.50       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.55       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.60       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.65       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.70       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.75       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
0.80       0.8741   0.3561   0.9325   0.9279   0.3481   0.3644  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8741, F1=0.3561, Normal Recall=0.9325, Normal Precision=0.9279, Attack Recall=0.3481, Attack Precision=0.3644

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
0.15       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662   <--
0.20       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.25       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.30       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.35       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.40       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.45       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.50       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.55       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.60       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.65       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.70       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.75       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
0.80       0.8164   0.4337   0.9327   0.8519   0.3515   0.5662  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8164, F1=0.4337, Normal Recall=0.9327, Normal Precision=0.8519, Attack Recall=0.3515, Attack Precision=0.5662

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
0.15       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932   <--
0.20       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.25       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.30       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.35       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.40       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.45       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.50       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.55       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.60       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.65       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.70       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.75       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
0.80       0.7588   0.4664   0.9333   0.7705   0.3515   0.6932  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7588, F1=0.4664, Normal Recall=0.9333, Normal Precision=0.7705, Attack Recall=0.3515, Attack Precision=0.6932

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
0.15       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770   <--
0.20       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.25       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.30       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.35       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.40       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.45       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.50       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.55       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.60       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.65       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.70       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.75       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
0.80       0.7002   0.4840   0.9328   0.6833   0.3515   0.7770  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7002, F1=0.4840, Normal Recall=0.9328, Normal Precision=0.6833, Attack Recall=0.3515, Attack Precision=0.7770

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
0.15       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409   <--
0.20       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.25       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.30       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.35       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.40       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.45       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.50       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.55       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.60       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.65       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.70       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.75       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
0.80       0.6425   0.4957   0.9335   0.5901   0.3515   0.8409  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6425, F1=0.4957, Normal Recall=0.9335, Normal Precision=0.5901, Attack Recall=0.3515, Attack Precision=0.8409

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
0.15       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873   <--
0.20       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.25       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.30       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.35       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.40       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.45       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.50       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.55       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.60       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.65       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.70       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.75       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
0.80       0.8956   0.6168   0.9018   0.9807   0.8402   0.4873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8956, F1=0.6168, Normal Recall=0.9018, Normal Precision=0.9807, Attack Recall=0.8402, Attack Precision=0.4873

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
0.15       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828   <--
0.20       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.25       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.30       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.35       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.40       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.45       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.50       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.55       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.60       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.65       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.70       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.75       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
0.80       0.8899   0.7529   0.9025   0.9574   0.8392   0.6828  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8899, F1=0.7529, Normal Recall=0.9025, Normal Precision=0.9574, Attack Recall=0.8392, Attack Precision=0.6828

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
0.15       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861   <--
0.20       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.25       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.30       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.35       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.40       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.45       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.50       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.55       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.60       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.65       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.70       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.75       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
0.80       0.8833   0.8118   0.9021   0.9290   0.8392   0.7861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8833, F1=0.8118, Normal Recall=0.9021, Normal Precision=0.9290, Attack Recall=0.8392, Attack Precision=0.7861

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
0.15       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512   <--
0.20       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.25       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.30       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.35       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.40       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.45       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.50       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.55       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.60       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.65       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.70       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.75       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
0.80       0.8770   0.8452   0.9022   0.8938   0.8392   0.8512  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8770, F1=0.8452, Normal Recall=0.9022, Normal Precision=0.8938, Attack Recall=0.8392, Attack Precision=0.8512

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
0.15       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938   <--
0.20       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.25       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.30       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.35       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.40       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.45       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.50       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.55       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.60       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.65       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.70       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.75       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
0.80       0.8698   0.8657   0.9003   0.8485   0.8392   0.8938  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8698, F1=0.8657, Normal Recall=0.9003, Normal Precision=0.8485, Attack Recall=0.8392, Attack Precision=0.8938

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
0.15       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891   <--
0.20       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.25       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.30       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.35       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.40       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.45       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.50       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.55       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.60       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.65       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.70       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.75       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.80       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8963, F1=0.6184, Normal Recall=0.9025, Normal Precision=0.9807, Attack Recall=0.8405, Attack Precision=0.4891

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
0.15       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842   <--
0.20       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.25       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.30       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.35       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.40       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.45       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.50       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.55       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.60       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.65       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.70       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.75       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.80       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8904, F1=0.7539, Normal Recall=0.9031, Normal Precision=0.9574, Attack Recall=0.8394, Attack Precision=0.6842

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
0.15       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872   <--
0.20       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.25       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.30       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.35       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.40       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.45       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.50       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.55       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.60       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.65       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.70       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.75       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.80       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8837, F1=0.8125, Normal Recall=0.9028, Normal Precision=0.9292, Attack Recall=0.8394, Attack Precision=0.7872

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
0.15       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520   <--
0.20       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.25       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.30       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.35       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.40       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.45       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.50       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.55       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.60       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.65       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.70       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.75       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.80       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8774, F1=0.8457, Normal Recall=0.9028, Normal Precision=0.8940, Attack Recall=0.8394, Attack Precision=0.8520

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
0.15       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943   <--
0.20       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.25       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.30       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.35       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.40       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.45       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.50       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.55       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.60       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.65       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.70       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.75       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.80       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8701, F1=0.8660, Normal Recall=0.9008, Normal Precision=0.8487, Attack Recall=0.8394, Attack Precision=0.8943

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
0.15       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891   <--
0.20       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.25       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.30       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.35       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.40       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.45       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.50       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.55       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.60       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.65       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.70       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.75       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
0.80       0.8963   0.6184   0.9025   0.9807   0.8405   0.4891  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8963, F1=0.6184, Normal Recall=0.9025, Normal Precision=0.9807, Attack Recall=0.8405, Attack Precision=0.4891

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
0.15       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842   <--
0.20       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.25       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.30       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.35       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.40       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.45       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.50       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.55       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.60       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.65       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.70       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.75       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
0.80       0.8904   0.7539   0.9031   0.9574   0.8394   0.6842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8904, F1=0.7539, Normal Recall=0.9031, Normal Precision=0.9574, Attack Recall=0.8394, Attack Precision=0.6842

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
0.15       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872   <--
0.20       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.25       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.30       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.35       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.40       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.45       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.50       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.55       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.60       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.65       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.70       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.75       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
0.80       0.8837   0.8125   0.9028   0.9292   0.8394   0.7872  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8837, F1=0.8125, Normal Recall=0.9028, Normal Precision=0.9292, Attack Recall=0.8394, Attack Precision=0.7872

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
0.15       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520   <--
0.20       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.25       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.30       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.35       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.40       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.45       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.50       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.55       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.60       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.65       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.70       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.75       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
0.80       0.8774   0.8457   0.9028   0.8940   0.8394   0.8520  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8774, F1=0.8457, Normal Recall=0.9028, Normal Precision=0.8940, Attack Recall=0.8394, Attack Precision=0.8520

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
0.15       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943   <--
0.20       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.25       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.30       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.35       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.40       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.45       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.50       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.55       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.60       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.65       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.70       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.75       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
0.80       0.8701   0.8660   0.9008   0.8487   0.8394   0.8943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8701, F1=0.8660, Normal Recall=0.9008, Normal Precision=0.8487, Attack Recall=0.8394, Attack Precision=0.8943

```

