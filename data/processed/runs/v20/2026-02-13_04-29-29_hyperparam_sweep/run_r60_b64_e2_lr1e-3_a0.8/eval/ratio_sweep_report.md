# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-14 06:24:45 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7793 | 0.7596 | 0.7409 | 0.7224 | 0.7056 | 0.6856 | 0.6673 | 0.6485 | 0.6298 | 0.6111 | 0.5928 |
| QAT+Prune only | 0.2669 | 0.3406 | 0.4138 | 0.4865 | 0.5607 | 0.6323 | 0.7066 | 0.7794 | 0.8518 | 0.9252 | 0.9981 |
| QAT+PTQ | 0.2676 | 0.3410 | 0.4141 | 0.4867 | 0.5609 | 0.6322 | 0.7068 | 0.7797 | 0.8519 | 0.9251 | 0.9981 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.2676 | 0.3410 | 0.4141 | 0.4867 | 0.5609 | 0.6322 | 0.7068 | 0.7797 | 0.8519 | 0.9251 | 0.9981 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3309 | 0.4779 | 0.5617 | 0.6170 | 0.6535 | 0.6813 | 0.7025 | 0.7193 | 0.7329 | 0.7444 |
| QAT+Prune only | 0.0000 | 0.2324 | 0.4051 | 0.5384 | 0.6451 | 0.7308 | 0.8032 | 0.8637 | 0.9151 | 0.9601 | 0.9991 |
| QAT+PTQ | 0.0000 | 0.2325 | 0.4053 | 0.5385 | 0.6452 | 0.7308 | 0.8033 | 0.8638 | 0.9152 | 0.9600 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2325 | 0.4053 | 0.5385 | 0.6452 | 0.7308 | 0.8033 | 0.8638 | 0.9152 | 0.9600 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7793 | 0.7780 | 0.7780 | 0.7779 | 0.7808 | 0.7783 | 0.7788 | 0.7785 | 0.7777 | 0.7754 | 0.0000 |
| QAT+Prune only | 0.2669 | 0.2675 | 0.2677 | 0.2673 | 0.2691 | 0.2665 | 0.2692 | 0.2691 | 0.2666 | 0.2691 | 0.0000 |
| QAT+PTQ | 0.2676 | 0.2680 | 0.2681 | 0.2675 | 0.2694 | 0.2664 | 0.2698 | 0.2700 | 0.2673 | 0.2681 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.2676 | 0.2680 | 0.2681 | 0.2675 | 0.2694 | 0.2664 | 0.2698 | 0.2700 | 0.2673 | 0.2681 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7793 | 0.0000 | 0.0000 | 0.0000 | 0.7793 | 1.0000 |
| 90 | 10 | 299,940 | 0.7596 | 0.2293 | 0.5945 | 0.3309 | 0.7780 | 0.9453 |
| 80 | 20 | 291,350 | 0.7409 | 0.4003 | 0.5928 | 0.4779 | 0.7780 | 0.8843 |
| 70 | 30 | 194,230 | 0.7224 | 0.5336 | 0.5929 | 0.5617 | 0.7779 | 0.8168 |
| 60 | 40 | 145,675 | 0.7056 | 0.6433 | 0.5928 | 0.6170 | 0.7808 | 0.7420 |
| 50 | 50 | 116,540 | 0.6856 | 0.7279 | 0.5928 | 0.6535 | 0.7783 | 0.6566 |
| 40 | 60 | 97,115 | 0.6673 | 0.8008 | 0.5929 | 0.6813 | 0.7788 | 0.5605 |
| 30 | 70 | 83,240 | 0.6485 | 0.8620 | 0.5928 | 0.7025 | 0.7785 | 0.4504 |
| 20 | 80 | 72,835 | 0.6298 | 0.9143 | 0.5928 | 0.7193 | 0.7777 | 0.3232 |
| 10 | 90 | 64,740 | 0.6111 | 0.9596 | 0.5929 | 0.7329 | 0.7754 | 0.1747 |
| 0 | 100 | 58,270 | 0.5928 | 1.0000 | 0.5928 | 0.7444 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2669 | 0.0000 | 0.0000 | 0.0000 | 0.2669 | 1.0000 |
| 90 | 10 | 299,940 | 0.3406 | 0.1315 | 0.9982 | 0.2324 | 0.2675 | 0.9993 |
| 80 | 20 | 291,350 | 0.4138 | 0.2541 | 0.9981 | 0.4051 | 0.2677 | 0.9983 |
| 70 | 30 | 194,230 | 0.4865 | 0.3686 | 0.9981 | 0.5384 | 0.2673 | 0.9970 |
| 60 | 40 | 145,675 | 0.5607 | 0.4765 | 0.9981 | 0.6451 | 0.2691 | 0.9954 |
| 50 | 50 | 116,540 | 0.6323 | 0.5764 | 0.9981 | 0.7308 | 0.2665 | 0.9931 |
| 40 | 60 | 97,115 | 0.7066 | 0.6720 | 0.9981 | 0.8032 | 0.2692 | 0.9898 |
| 30 | 70 | 83,240 | 0.7794 | 0.7611 | 0.9981 | 0.8637 | 0.2691 | 0.9842 |
| 20 | 80 | 72,835 | 0.8518 | 0.8448 | 0.9981 | 0.9151 | 0.2666 | 0.9729 |
| 10 | 90 | 64,740 | 0.9252 | 0.9248 | 0.9981 | 0.9601 | 0.2691 | 0.9416 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2676 | 0.0000 | 0.0000 | 0.0000 | 0.2676 | 1.0000 |
| 90 | 10 | 299,940 | 0.3410 | 0.1316 | 0.9982 | 0.2325 | 0.2680 | 0.9993 |
| 80 | 20 | 291,350 | 0.4141 | 0.2542 | 0.9981 | 0.4053 | 0.2681 | 0.9982 |
| 70 | 30 | 194,230 | 0.4867 | 0.3687 | 0.9981 | 0.5385 | 0.2675 | 0.9970 |
| 60 | 40 | 145,675 | 0.5609 | 0.4766 | 0.9981 | 0.6452 | 0.2694 | 0.9953 |
| 50 | 50 | 116,540 | 0.6322 | 0.5764 | 0.9981 | 0.7308 | 0.2664 | 0.9929 |
| 40 | 60 | 97,115 | 0.7068 | 0.6722 | 0.9981 | 0.8033 | 0.2698 | 0.9895 |
| 30 | 70 | 83,240 | 0.7797 | 0.7613 | 0.9981 | 0.8638 | 0.2700 | 0.9838 |
| 20 | 80 | 72,835 | 0.8519 | 0.8449 | 0.9981 | 0.9152 | 0.2673 | 0.9723 |
| 10 | 90 | 64,740 | 0.9251 | 0.9247 | 0.9981 | 0.9600 | 0.2681 | 0.9399 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.2676 | 0.0000 | 0.0000 | 0.0000 | 0.2676 | 1.0000 |
| 90 | 10 | 299,940 | 0.3410 | 0.1316 | 0.9982 | 0.2325 | 0.2680 | 0.9993 |
| 80 | 20 | 291,350 | 0.4141 | 0.2542 | 0.9981 | 0.4053 | 0.2681 | 0.9982 |
| 70 | 30 | 194,230 | 0.4867 | 0.3687 | 0.9981 | 0.5385 | 0.2675 | 0.9970 |
| 60 | 40 | 145,675 | 0.5609 | 0.4766 | 0.9981 | 0.6452 | 0.2694 | 0.9953 |
| 50 | 50 | 116,540 | 0.6322 | 0.5764 | 0.9981 | 0.7308 | 0.2664 | 0.9929 |
| 40 | 60 | 97,115 | 0.7068 | 0.6722 | 0.9981 | 0.8033 | 0.2698 | 0.9895 |
| 30 | 70 | 83,240 | 0.7797 | 0.7613 | 0.9981 | 0.8638 | 0.2700 | 0.9838 |
| 20 | 80 | 72,835 | 0.8519 | 0.8449 | 0.9981 | 0.9152 | 0.2673 | 0.9723 |
| 10 | 90 | 64,740 | 0.9251 | 0.9247 | 0.9981 | 0.9600 | 0.2681 | 0.9399 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293   <--
0.20       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.25       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.30       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.35       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.40       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.45       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.50       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.55       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.60       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.65       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.70       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.75       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
0.80       0.7596   0.3310   0.7780   0.9453   0.5947   0.2293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7596, F1=0.3310, Normal Recall=0.7780, Normal Precision=0.9453, Attack Recall=0.5947, Attack Precision=0.2293

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
0.15       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001   <--
0.20       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.25       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.30       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.35       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.40       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.45       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.50       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.55       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.60       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.65       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.70       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.75       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
0.80       0.7408   0.4778   0.7778   0.8843   0.5928   0.4001  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7408, F1=0.4778, Normal Recall=0.7778, Normal Precision=0.8843, Attack Recall=0.5928, Attack Precision=0.4001

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
0.15       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345   <--
0.20       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.25       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.30       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.35       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.40       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.45       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.50       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.55       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.60       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.65       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.70       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.75       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
0.80       0.7230   0.5622   0.7787   0.8169   0.5929   0.5345  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7230, F1=0.5622, Normal Recall=0.7787, Normal Precision=0.8169, Attack Recall=0.5929, Attack Precision=0.5345

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
0.15       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415   <--
0.20       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.25       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.30       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.35       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.40       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.45       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.50       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.55       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.60       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.65       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.70       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.75       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
0.80       0.7046   0.6162   0.7792   0.7416   0.5928   0.6415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7046, F1=0.6162, Normal Recall=0.7792, Normal Precision=0.7416, Attack Recall=0.5928, Attack Precision=0.6415

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
0.15       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296   <--
0.20       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.25       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.30       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.35       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.40       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.45       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.50       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.55       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.60       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.65       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.70       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.75       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
0.80       0.6866   0.6541   0.7803   0.6571   0.5928   0.7296  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6866, F1=0.6541, Normal Recall=0.7803, Normal Precision=0.6571, Attack Recall=0.5928, Attack Precision=0.7296

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
0.15       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315   <--
0.20       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.25       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.30       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.35       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.40       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.45       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.50       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.55       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.60       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.65       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.70       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.75       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
0.80       0.3406   0.2324   0.2675   0.9993   0.9984   0.1315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3406, F1=0.2324, Normal Recall=0.2675, Normal Precision=0.9993, Attack Recall=0.9984, Attack Precision=0.1315

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
0.15       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541   <--
0.20       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.25       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.30       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.35       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.40       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.45       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.50       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.55       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.60       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.65       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.70       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.75       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
0.80       0.4135   0.4050   0.2673   0.9983   0.9981   0.2541  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4135, F1=0.4050, Normal Recall=0.2673, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.2541

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
0.15       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689   <--
0.20       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.25       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.30       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.35       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.40       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.45       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.50       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.55       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.60       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.65       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.70       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.75       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
0.80       0.4871   0.5387   0.2681   0.9970   0.9981   0.3689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4871, F1=0.5387, Normal Recall=0.2681, Normal Precision=0.9970, Attack Recall=0.9981, Attack Precision=0.3689

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
0.15       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760   <--
0.20       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.25       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.30       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.35       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.40       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.45       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.50       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.55       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.60       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.65       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.70       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.75       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
0.80       0.5597   0.6446   0.2674   0.9954   0.9981   0.4760  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5597, F1=0.6446, Normal Recall=0.2674, Normal Precision=0.9954, Attack Recall=0.9981, Attack Precision=0.4760

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
0.15       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764   <--
0.20       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.25       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.30       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.35       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.40       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.45       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.50       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.55       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.60       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.65       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.70       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.75       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
0.80       0.6323   0.7308   0.2665   0.9931   0.9981   0.5764  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6323, F1=0.7308, Normal Recall=0.2665, Normal Precision=0.9931, Attack Recall=0.9981, Attack Precision=0.5764

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
0.15       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316   <--
0.20       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.25       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.30       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.35       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.40       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.45       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.50       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.55       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.60       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.65       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.70       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.75       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.80       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3410, F1=0.2325, Normal Recall=0.2680, Normal Precision=0.9993, Attack Recall=0.9983, Attack Precision=0.1316

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
0.15       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542   <--
0.20       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.25       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.30       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.35       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.40       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.45       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.50       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.55       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.60       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.65       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.70       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.75       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.80       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4138, F1=0.4051, Normal Recall=0.2677, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.2542

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
0.15       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690   <--
0.20       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.25       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.30       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.35       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.40       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.45       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.50       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.55       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.60       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.65       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.70       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.75       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.80       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4875, F1=0.5388, Normal Recall=0.2686, Normal Precision=0.9970, Attack Recall=0.9981, Attack Precision=0.3690

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
0.15       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761   <--
0.20       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.25       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.30       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.35       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.40       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.45       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.50       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.55       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.60       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.65       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.70       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.75       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.80       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5599, F1=0.6447, Normal Recall=0.2678, Normal Precision=0.9953, Attack Recall=0.9981, Attack Precision=0.4761

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
0.15       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765   <--
0.20       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.25       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.30       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.35       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.40       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.45       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.50       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.55       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.60       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.65       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.70       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.75       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.80       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6325, F1=0.7309, Normal Recall=0.2669, Normal Precision=0.9929, Attack Recall=0.9981, Attack Precision=0.5765

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
0.15       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316   <--
0.20       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.25       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.30       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.35       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.40       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.45       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.50       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.55       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.60       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.65       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.70       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.75       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
0.80       0.3410   0.2325   0.2680   0.9993   0.9983   0.1316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3410, F1=0.2325, Normal Recall=0.2680, Normal Precision=0.9993, Attack Recall=0.9983, Attack Precision=0.1316

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
0.15       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542   <--
0.20       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.25       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.30       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.35       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.40       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.45       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.50       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.55       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.60       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.65       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.70       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.75       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
0.80       0.4138   0.4051   0.2677   0.9982   0.9981   0.2542  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4138, F1=0.4051, Normal Recall=0.2677, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.2542

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
0.15       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690   <--
0.20       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.25       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.30       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.35       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.40       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.45       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.50       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.55       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.60       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.65       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.70       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.75       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
0.80       0.4875   0.5388   0.2686   0.9970   0.9981   0.3690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4875, F1=0.5388, Normal Recall=0.2686, Normal Precision=0.9970, Attack Recall=0.9981, Attack Precision=0.3690

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
0.15       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761   <--
0.20       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.25       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.30       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.35       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.40       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.45       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.50       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.55       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.60       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.65       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.70       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.75       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
0.80       0.5599   0.6447   0.2678   0.9953   0.9981   0.4761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5599, F1=0.6447, Normal Recall=0.2678, Normal Precision=0.9953, Attack Recall=0.9981, Attack Precision=0.4761

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
0.15       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765   <--
0.20       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.25       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.30       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.35       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.40       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.45       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.50       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.55       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.60       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.65       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.70       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.75       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
0.80       0.6325   0.7309   0.2669   0.9929   0.9981   0.5765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6325, F1=0.7309, Normal Recall=0.2669, Normal Precision=0.9929, Attack Recall=0.9981, Attack Precision=0.5765

```

