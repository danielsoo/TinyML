# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-18 13:01:28 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7752 | 0.7970 | 0.8191 | 0.8421 | 0.8641 | 0.8855 | 0.9090 | 0.9324 | 0.9543 | 0.9757 | 0.9988 |
| QAT+Prune only | 0.7022 | 0.7312 | 0.7600 | 0.7914 | 0.8202 | 0.8487 | 0.8792 | 0.9090 | 0.9386 | 0.9677 | 0.9977 |
| QAT+PTQ | 0.7016 | 0.7308 | 0.7597 | 0.7911 | 0.8199 | 0.8486 | 0.8790 | 0.9088 | 0.9385 | 0.9678 | 0.9977 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7016 | 0.7308 | 0.7597 | 0.7911 | 0.8199 | 0.8486 | 0.8790 | 0.9088 | 0.9385 | 0.9678 | 0.9977 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4960 | 0.6884 | 0.7915 | 0.8546 | 0.8971 | 0.9294 | 0.9539 | 0.9722 | 0.9867 | 0.9994 |
| QAT+Prune only | 0.0000 | 0.4260 | 0.6245 | 0.7416 | 0.8161 | 0.8683 | 0.9084 | 0.9388 | 0.9630 | 0.9823 | 0.9988 |
| QAT+PTQ | 0.0000 | 0.4257 | 0.6242 | 0.7413 | 0.8159 | 0.8682 | 0.9082 | 0.9387 | 0.9629 | 0.9824 | 0.9989 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4257 | 0.6242 | 0.7413 | 0.8159 | 0.8682 | 0.9082 | 0.9387 | 0.9629 | 0.9824 | 0.9989 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7752 | 0.7745 | 0.7742 | 0.7750 | 0.7743 | 0.7722 | 0.7742 | 0.7774 | 0.7762 | 0.7686 | 0.0000 |
| QAT+Prune only | 0.7022 | 0.7016 | 0.7006 | 0.7030 | 0.7019 | 0.6998 | 0.7015 | 0.7021 | 0.7022 | 0.6980 | 0.0000 |
| QAT+PTQ | 0.7016 | 0.7011 | 0.7002 | 0.7025 | 0.7014 | 0.6994 | 0.7010 | 0.7014 | 0.7015 | 0.6985 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7016 | 0.7011 | 0.7002 | 0.7025 | 0.7014 | 0.6994 | 0.7010 | 0.7014 | 0.7015 | 0.6985 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7752 | 0.0000 | 0.0000 | 0.0000 | 0.7752 | 1.0000 |
| 90 | 10 | 299,940 | 0.7970 | 0.3299 | 0.9991 | 0.4960 | 0.7745 | 0.9999 |
| 80 | 20 | 291,350 | 0.8191 | 0.5252 | 0.9988 | 0.6884 | 0.7742 | 0.9996 |
| 70 | 30 | 194,230 | 0.8421 | 0.6554 | 0.9988 | 0.7915 | 0.7750 | 0.9993 |
| 60 | 40 | 145,675 | 0.8641 | 0.7469 | 0.9988 | 0.8546 | 0.7743 | 0.9989 |
| 50 | 50 | 116,540 | 0.8855 | 0.8143 | 0.9988 | 0.8971 | 0.7722 | 0.9984 |
| 40 | 60 | 97,115 | 0.9090 | 0.8690 | 0.9988 | 0.9294 | 0.7742 | 0.9976 |
| 30 | 70 | 83,240 | 0.9324 | 0.9128 | 0.9988 | 0.9539 | 0.7774 | 0.9963 |
| 20 | 80 | 72,835 | 0.9543 | 0.9470 | 0.9988 | 0.9722 | 0.7762 | 0.9937 |
| 10 | 90 | 64,740 | 0.9757 | 0.9749 | 0.9988 | 0.9867 | 0.7686 | 0.9857 |
| 0 | 100 | 58,270 | 0.9988 | 1.0000 | 0.9988 | 0.9994 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7022 | 0.0000 | 0.0000 | 0.0000 | 0.7022 | 1.0000 |
| 90 | 10 | 299,940 | 0.7312 | 0.2708 | 0.9976 | 0.4260 | 0.7016 | 0.9996 |
| 80 | 20 | 291,350 | 0.7600 | 0.4545 | 0.9977 | 0.6245 | 0.7006 | 0.9992 |
| 70 | 30 | 194,230 | 0.7914 | 0.5901 | 0.9977 | 0.7416 | 0.7030 | 0.9986 |
| 60 | 40 | 145,675 | 0.8202 | 0.6905 | 0.9977 | 0.8161 | 0.7019 | 0.9978 |
| 50 | 50 | 116,540 | 0.8487 | 0.7687 | 0.9977 | 0.8683 | 0.6998 | 0.9967 |
| 40 | 60 | 97,115 | 0.8792 | 0.8337 | 0.9977 | 0.9084 | 0.7015 | 0.9951 |
| 30 | 70 | 83,240 | 0.9090 | 0.8865 | 0.9977 | 0.9388 | 0.7021 | 0.9924 |
| 20 | 80 | 72,835 | 0.9386 | 0.9306 | 0.9977 | 0.9630 | 0.7022 | 0.9871 |
| 10 | 90 | 64,740 | 0.9677 | 0.9675 | 0.9977 | 0.9823 | 0.6980 | 0.9712 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9988 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7016 | 0.0000 | 0.0000 | 0.0000 | 0.7016 | 1.0000 |
| 90 | 10 | 299,940 | 0.7308 | 0.2706 | 0.9977 | 0.4257 | 0.7011 | 0.9996 |
| 80 | 20 | 291,350 | 0.7597 | 0.4541 | 0.9977 | 0.6242 | 0.7002 | 0.9992 |
| 70 | 30 | 194,230 | 0.7911 | 0.5897 | 0.9977 | 0.7413 | 0.7025 | 0.9986 |
| 60 | 40 | 145,675 | 0.8199 | 0.6902 | 0.9977 | 0.8159 | 0.7014 | 0.9978 |
| 50 | 50 | 116,540 | 0.8486 | 0.7685 | 0.9977 | 0.8682 | 0.6994 | 0.9967 |
| 40 | 60 | 97,115 | 0.8790 | 0.8335 | 0.9977 | 0.9082 | 0.7010 | 0.9951 |
| 30 | 70 | 83,240 | 0.9088 | 0.8863 | 0.9977 | 0.9387 | 0.7014 | 0.9925 |
| 20 | 80 | 72,835 | 0.9385 | 0.9304 | 0.9977 | 0.9629 | 0.7015 | 0.9872 |
| 10 | 90 | 64,740 | 0.9678 | 0.9675 | 0.9977 | 0.9824 | 0.6985 | 0.9714 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9989 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7016 | 0.0000 | 0.0000 | 0.0000 | 0.7016 | 1.0000 |
| 90 | 10 | 299,940 | 0.7308 | 0.2706 | 0.9977 | 0.4257 | 0.7011 | 0.9996 |
| 80 | 20 | 291,350 | 0.7597 | 0.4541 | 0.9977 | 0.6242 | 0.7002 | 0.9992 |
| 70 | 30 | 194,230 | 0.7911 | 0.5897 | 0.9977 | 0.7413 | 0.7025 | 0.9986 |
| 60 | 40 | 145,675 | 0.8199 | 0.6902 | 0.9977 | 0.8159 | 0.7014 | 0.9978 |
| 50 | 50 | 116,540 | 0.8486 | 0.7685 | 0.9977 | 0.8682 | 0.6994 | 0.9967 |
| 40 | 60 | 97,115 | 0.8790 | 0.8335 | 0.9977 | 0.9082 | 0.7010 | 0.9951 |
| 30 | 70 | 83,240 | 0.9088 | 0.8863 | 0.9977 | 0.9387 | 0.7014 | 0.9925 |
| 20 | 80 | 72,835 | 0.9385 | 0.9304 | 0.9977 | 0.9629 | 0.7015 | 0.9872 |
| 10 | 90 | 64,740 | 0.9678 | 0.9675 | 0.9977 | 0.9824 | 0.6985 | 0.9714 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9989 | 0.0000 | 0.0000 |


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
0.15       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298   <--
0.20       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.25       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.30       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.35       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.40       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.45       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.50       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.55       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.60       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.65       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.70       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.75       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
0.80       0.7969   0.4959   0.7745   0.9998   0.9989   0.3298  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7969, F1=0.4959, Normal Recall=0.7745, Normal Precision=0.9998, Attack Recall=0.9989, Attack Precision=0.3298

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
0.15       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263   <--
0.20       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.25       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.30       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.35       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.40       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.45       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.50       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.55       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.60       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.65       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.70       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.75       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
0.80       0.8199   0.6893   0.7752   0.9996   0.9988   0.5263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8199, F1=0.6893, Normal Recall=0.7752, Normal Precision=0.9996, Attack Recall=0.9988, Attack Precision=0.5263

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
0.15       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558   <--
0.20       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.25       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.30       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.35       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.40       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.45       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.50       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.55       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.60       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.65       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.70       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.75       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
0.80       0.8424   0.7917   0.7753   0.9993   0.9988   0.6558  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8424, F1=0.7917, Normal Recall=0.7753, Normal Precision=0.9993, Attack Recall=0.9988, Attack Precision=0.6558

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
0.15       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481   <--
0.20       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.25       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.30       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.35       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.40       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.45       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.50       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.55       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.60       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.65       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.70       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.75       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
0.80       0.8650   0.8554   0.7757   0.9989   0.9988   0.7481  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8650, F1=0.8554, Normal Recall=0.7757, Normal Precision=0.9989, Attack Recall=0.9988, Attack Precision=0.7481

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
0.15       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162   <--
0.20       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.25       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.30       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.35       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.40       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.45       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.50       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.55       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.60       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.65       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.70       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.75       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
0.80       0.8870   0.8983   0.7752   0.9984   0.9988   0.8162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8870, F1=0.8983, Normal Recall=0.7752, Normal Precision=0.9984, Attack Recall=0.9988, Attack Precision=0.8162

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
0.15       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708   <--
0.20       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.25       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.30       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.35       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.40       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.45       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.50       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.55       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.60       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.65       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.70       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.75       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
0.80       0.7312   0.4260   0.7016   0.9996   0.9977   0.2708  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7312, F1=0.4260, Normal Recall=0.7016, Normal Precision=0.9996, Attack Recall=0.9977, Attack Precision=0.2708

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
0.15       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561   <--
0.20       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.25       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.30       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.35       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.40       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.45       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.50       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.55       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.60       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.65       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.70       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.75       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
0.80       0.7616   0.6261   0.7026   0.9992   0.9977   0.4561  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7616, F1=0.6261, Normal Recall=0.7026, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4561

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
0.15       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893   <--
0.20       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.25       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.30       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.35       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.40       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.45       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.50       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.55       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.60       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.65       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.70       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.75       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
0.80       0.7907   0.7410   0.7020   0.9986   0.9977   0.5893  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7907, F1=0.7410, Normal Recall=0.7020, Normal Precision=0.9986, Attack Recall=0.9977, Attack Precision=0.5893

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
0.15       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905   <--
0.20       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.25       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.30       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.35       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.40       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.45       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.50       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.55       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.60       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.65       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.70       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.75       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
0.80       0.8202   0.8161   0.7018   0.9978   0.9977   0.6905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8202, F1=0.8161, Normal Recall=0.7018, Normal Precision=0.9978, Attack Recall=0.9977, Attack Precision=0.6905

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
0.15       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691   <--
0.20       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.25       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.30       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.35       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.40       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.45       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.50       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.55       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.60       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.65       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.70       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.75       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
0.80       0.8491   0.8686   0.7005   0.9967   0.9977   0.7691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8491, F1=0.8686, Normal Recall=0.7005, Normal Precision=0.9967, Attack Recall=0.9977, Attack Precision=0.7691

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
0.15       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706   <--
0.20       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.25       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.30       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.35       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.40       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.45       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.50       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.55       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.60       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.65       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.70       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.75       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.80       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7308, F1=0.4257, Normal Recall=0.7011, Normal Precision=0.9996, Attack Recall=0.9977, Attack Precision=0.2706

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
0.15       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557   <--
0.20       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.25       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.30       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.35       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.40       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.45       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.50       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.55       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.60       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.65       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.70       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.75       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.80       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7612, F1=0.6257, Normal Recall=0.7021, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4557

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
0.15       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888   <--
0.20       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.25       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.30       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.35       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.40       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.45       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.50       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.55       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.60       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.65       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.70       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.75       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.80       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7903, F1=0.7405, Normal Recall=0.7013, Normal Precision=0.9986, Attack Recall=0.9977, Attack Precision=0.5888

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
0.15       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901   <--
0.20       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.25       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.30       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.35       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.40       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.45       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.50       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.55       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.60       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.65       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.70       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.75       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.80       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8199, F1=0.8159, Normal Recall=0.7013, Normal Precision=0.9978, Attack Recall=0.9977, Attack Precision=0.6901

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
0.15       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690   <--
0.20       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.25       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.30       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.35       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.40       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.45       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.50       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.55       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.60       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.65       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.70       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.75       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.80       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8490, F1=0.8686, Normal Recall=0.7003, Normal Precision=0.9968, Attack Recall=0.9977, Attack Precision=0.7690

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
0.15       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706   <--
0.20       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.25       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.30       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.35       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.40       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.45       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.50       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.55       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.60       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.65       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.70       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.75       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
0.80       0.7308   0.4257   0.7011   0.9996   0.9977   0.2706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7308, F1=0.4257, Normal Recall=0.7011, Normal Precision=0.9996, Attack Recall=0.9977, Attack Precision=0.2706

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
0.15       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557   <--
0.20       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.25       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.30       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.35       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.40       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.45       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.50       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.55       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.60       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.65       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.70       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.75       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
0.80       0.7612   0.6257   0.7021   0.9992   0.9977   0.4557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7612, F1=0.6257, Normal Recall=0.7021, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4557

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
0.15       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888   <--
0.20       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.25       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.30       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.35       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.40       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.45       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.50       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.55       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.60       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.65       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.70       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.75       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
0.80       0.7903   0.7405   0.7013   0.9986   0.9977   0.5888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7903, F1=0.7405, Normal Recall=0.7013, Normal Precision=0.9986, Attack Recall=0.9977, Attack Precision=0.5888

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
0.15       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901   <--
0.20       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.25       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.30       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.35       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.40       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.45       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.50       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.55       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.60       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.65       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.70       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.75       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
0.80       0.8199   0.8159   0.7013   0.9978   0.9977   0.6901  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8199, F1=0.8159, Normal Recall=0.7013, Normal Precision=0.9978, Attack Recall=0.9977, Attack Precision=0.6901

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
0.15       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690   <--
0.20       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.25       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.30       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.35       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.40       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.45       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.50       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.55       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.60       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.65       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.70       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.75       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
0.80       0.8490   0.8686   0.7003   0.9968   0.9977   0.7690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8490, F1=0.8686, Normal Recall=0.7003, Normal Precision=0.9968, Attack Recall=0.9977, Attack Precision=0.7690

```

