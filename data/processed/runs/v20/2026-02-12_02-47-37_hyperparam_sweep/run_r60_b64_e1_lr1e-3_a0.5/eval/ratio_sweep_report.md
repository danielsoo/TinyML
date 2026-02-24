# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-12 11:56:38 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6537 | 0.6862 | 0.7207 | 0.7562 | 0.7903 | 0.8245 | 0.8598 | 0.8953 | 0.9300 | 0.9648 | 0.9996 |
| QAT+Prune only | 0.7804 | 0.8021 | 0.8233 | 0.8455 | 0.8668 | 0.8876 | 0.9098 | 0.9323 | 0.9532 | 0.9751 | 0.9968 |
| QAT+PTQ | 0.7786 | 0.8002 | 0.8216 | 0.8440 | 0.8657 | 0.8864 | 0.9090 | 0.9318 | 0.9527 | 0.9748 | 0.9967 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7786 | 0.8002 | 0.8216 | 0.8440 | 0.8657 | 0.8864 | 0.9090 | 0.9318 | 0.9527 | 0.9748 | 0.9967 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3891 | 0.5887 | 0.7110 | 0.7923 | 0.8506 | 0.8954 | 0.9304 | 0.9581 | 0.9808 | 0.9998 |
| QAT+Prune only | 0.0000 | 0.5019 | 0.6929 | 0.7947 | 0.8569 | 0.8987 | 0.9299 | 0.9538 | 0.9715 | 0.9863 | 0.9984 |
| QAT+PTQ | 0.0000 | 0.4995 | 0.6908 | 0.7931 | 0.8558 | 0.8977 | 0.9293 | 0.9534 | 0.9712 | 0.9861 | 0.9983 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4995 | 0.6908 | 0.7931 | 0.8558 | 0.8977 | 0.9293 | 0.9534 | 0.9712 | 0.9861 | 0.9983 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6537 | 0.6513 | 0.6509 | 0.6519 | 0.6508 | 0.6494 | 0.6502 | 0.6519 | 0.6516 | 0.6514 | 0.0000 |
| QAT+Prune only | 0.7804 | 0.7804 | 0.7799 | 0.7807 | 0.7802 | 0.7784 | 0.7792 | 0.7819 | 0.7790 | 0.7797 | 0.0000 |
| QAT+PTQ | 0.7786 | 0.7783 | 0.7778 | 0.7785 | 0.7784 | 0.7762 | 0.7775 | 0.7803 | 0.7770 | 0.7779 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7786 | 0.7783 | 0.7778 | 0.7785 | 0.7784 | 0.7762 | 0.7775 | 0.7803 | 0.7770 | 0.7779 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6537 | 0.0000 | 0.0000 | 0.0000 | 0.6537 | 1.0000 |
| 90 | 10 | 299,940 | 0.6862 | 0.2416 | 0.9996 | 0.3891 | 0.6513 | 0.9999 |
| 80 | 20 | 291,350 | 0.7207 | 0.4172 | 0.9996 | 0.5887 | 0.6509 | 0.9998 |
| 70 | 30 | 194,230 | 0.7562 | 0.5517 | 0.9996 | 0.7110 | 0.6519 | 0.9997 |
| 60 | 40 | 145,675 | 0.7903 | 0.6562 | 0.9996 | 0.7923 | 0.6508 | 0.9996 |
| 50 | 50 | 116,540 | 0.8245 | 0.7403 | 0.9996 | 0.8506 | 0.6494 | 0.9994 |
| 40 | 60 | 97,115 | 0.8598 | 0.8108 | 0.9996 | 0.8954 | 0.6502 | 0.9991 |
| 30 | 70 | 83,240 | 0.8953 | 0.8701 | 0.9996 | 0.9304 | 0.6519 | 0.9985 |
| 20 | 80 | 72,835 | 0.9300 | 0.9199 | 0.9996 | 0.9581 | 0.6516 | 0.9975 |
| 10 | 90 | 64,740 | 0.9648 | 0.9627 | 0.9996 | 0.9808 | 0.6514 | 0.9943 |
| 0 | 100 | 58,270 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7804 | 0.0000 | 0.0000 | 0.0000 | 0.7804 | 1.0000 |
| 90 | 10 | 299,940 | 0.8021 | 0.3353 | 0.9970 | 0.5019 | 0.7804 | 0.9996 |
| 80 | 20 | 291,350 | 0.8233 | 0.5310 | 0.9968 | 0.6929 | 0.7799 | 0.9990 |
| 70 | 30 | 194,230 | 0.8455 | 0.6608 | 0.9968 | 0.7947 | 0.7807 | 0.9982 |
| 60 | 40 | 145,675 | 0.8668 | 0.7515 | 0.9968 | 0.8569 | 0.7802 | 0.9973 |
| 50 | 50 | 116,540 | 0.8876 | 0.8181 | 0.9968 | 0.8987 | 0.7784 | 0.9959 |
| 40 | 60 | 97,115 | 0.9098 | 0.8713 | 0.9968 | 0.9299 | 0.7792 | 0.9939 |
| 30 | 70 | 83,240 | 0.9323 | 0.9143 | 0.9968 | 0.9538 | 0.7819 | 0.9905 |
| 20 | 80 | 72,835 | 0.9532 | 0.9475 | 0.9968 | 0.9715 | 0.7790 | 0.9838 |
| 10 | 90 | 64,740 | 0.9751 | 0.9760 | 0.9968 | 0.9863 | 0.7797 | 0.9643 |
| 0 | 100 | 58,270 | 0.9968 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7786 | 0.0000 | 0.0000 | 0.0000 | 0.7786 | 1.0000 |
| 90 | 10 | 299,940 | 0.8002 | 0.3332 | 0.9969 | 0.4995 | 0.7783 | 0.9996 |
| 80 | 20 | 291,350 | 0.8216 | 0.5286 | 0.9967 | 0.6908 | 0.7778 | 0.9989 |
| 70 | 30 | 194,230 | 0.8440 | 0.6585 | 0.9967 | 0.7931 | 0.7785 | 0.9982 |
| 60 | 40 | 145,675 | 0.8657 | 0.7499 | 0.9967 | 0.8558 | 0.7784 | 0.9971 |
| 50 | 50 | 116,540 | 0.8864 | 0.8166 | 0.9967 | 0.8977 | 0.7762 | 0.9957 |
| 40 | 60 | 97,115 | 0.9090 | 0.8705 | 0.9967 | 0.9293 | 0.7775 | 0.9936 |
| 30 | 70 | 83,240 | 0.9318 | 0.9137 | 0.9967 | 0.9534 | 0.7803 | 0.9901 |
| 20 | 80 | 72,835 | 0.9527 | 0.9470 | 0.9967 | 0.9712 | 0.7770 | 0.9831 |
| 10 | 90 | 64,740 | 0.9748 | 0.9758 | 0.9967 | 0.9861 | 0.7779 | 0.9627 |
| 0 | 100 | 58,270 | 0.9967 | 1.0000 | 0.9967 | 0.9983 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7786 | 0.0000 | 0.0000 | 0.0000 | 0.7786 | 1.0000 |
| 90 | 10 | 299,940 | 0.8002 | 0.3332 | 0.9969 | 0.4995 | 0.7783 | 0.9996 |
| 80 | 20 | 291,350 | 0.8216 | 0.5286 | 0.9967 | 0.6908 | 0.7778 | 0.9989 |
| 70 | 30 | 194,230 | 0.8440 | 0.6585 | 0.9967 | 0.7931 | 0.7785 | 0.9982 |
| 60 | 40 | 145,675 | 0.8657 | 0.7499 | 0.9967 | 0.8558 | 0.7784 | 0.9971 |
| 50 | 50 | 116,540 | 0.8864 | 0.8166 | 0.9967 | 0.8977 | 0.7762 | 0.9957 |
| 40 | 60 | 97,115 | 0.9090 | 0.8705 | 0.9967 | 0.9293 | 0.7775 | 0.9936 |
| 30 | 70 | 83,240 | 0.9318 | 0.9137 | 0.9967 | 0.9534 | 0.7803 | 0.9901 |
| 20 | 80 | 72,835 | 0.9527 | 0.9470 | 0.9967 | 0.9712 | 0.7770 | 0.9831 |
| 10 | 90 | 64,740 | 0.9748 | 0.9758 | 0.9967 | 0.9861 | 0.7779 | 0.9627 |
| 0 | 100 | 58,270 | 0.9967 | 1.0000 | 0.9967 | 0.9983 | 0.0000 | 0.0000 |


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
0.15       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416   <--
0.20       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.25       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.30       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.35       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.40       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.45       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.50       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.55       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.60       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.65       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.70       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.75       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
0.80       0.6862   0.3891   0.6513   0.9999   0.9996   0.2416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6862, F1=0.3891, Normal Recall=0.6513, Normal Precision=0.9999, Attack Recall=0.9996, Attack Precision=0.2416

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
0.15       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181   <--
0.20       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.25       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.30       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.35       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.40       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.45       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.50       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.55       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.60       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.65       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.70       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.75       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
0.80       0.7217   0.5896   0.6522   0.9998   0.9996   0.4181  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7217, F1=0.5896, Normal Recall=0.6522, Normal Precision=0.9998, Attack Recall=0.9996, Attack Precision=0.4181

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
0.15       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528   <--
0.20       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.25       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.30       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.35       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.40       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.45       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.50       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.55       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.60       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.65       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.70       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.75       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
0.80       0.7573   0.7119   0.6534   0.9997   0.9996   0.5528  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7573, F1=0.7119, Normal Recall=0.6534, Normal Precision=0.9997, Attack Recall=0.9996, Attack Precision=0.5528

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
0.15       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576   <--
0.20       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.25       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.30       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.35       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.40       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.45       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.50       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.55       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.60       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.65       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.70       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.75       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
0.80       0.7916   0.7933   0.6530   0.9996   0.9996   0.6576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7916, F1=0.7933, Normal Recall=0.6530, Normal Precision=0.9996, Attack Recall=0.9996, Attack Precision=0.6576

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
0.15       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423   <--
0.20       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.25       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.30       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.35       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.40       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.45       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.50       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.55       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.60       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.65       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.70       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.75       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
0.80       0.8262   0.8519   0.6529   0.9994   0.9996   0.7423  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8262, F1=0.8519, Normal Recall=0.6529, Normal Precision=0.9994, Attack Recall=0.9996, Attack Precision=0.7423

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
0.15       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353   <--
0.20       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.25       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.30       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.35       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.40       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.45       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.50       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.55       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.60       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.65       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.70       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.75       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
0.80       0.8021   0.5018   0.7804   0.9996   0.9969   0.3353  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8021, F1=0.5018, Normal Recall=0.7804, Normal Precision=0.9996, Attack Recall=0.9969, Attack Precision=0.3353

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
0.15       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322   <--
0.20       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.25       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.30       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.35       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.40       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.45       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.50       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.55       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.60       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.65       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.70       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.75       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
0.80       0.8242   0.6939   0.7810   0.9990   0.9968   0.5322  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8242, F1=0.6939, Normal Recall=0.7810, Normal Precision=0.9990, Attack Recall=0.9968, Attack Precision=0.5322

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
0.15       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611   <--
0.20       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.25       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.30       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.35       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.40       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.45       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.50       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.55       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.60       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.65       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.70       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.75       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
0.80       0.8457   0.7949   0.7810   0.9982   0.9968   0.6611  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8457, F1=0.7949, Normal Recall=0.7810, Normal Precision=0.9982, Attack Recall=0.9968, Attack Precision=0.6611

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
0.15       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515   <--
0.20       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.25       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.30       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.35       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.40       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.45       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.50       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.55       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.60       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.65       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.70       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.75       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
0.80       0.8668   0.8569   0.7802   0.9973   0.9968   0.7515  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8668, F1=0.8569, Normal Recall=0.7802, Normal Precision=0.9973, Attack Recall=0.9968, Attack Precision=0.7515

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
0.15       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177   <--
0.20       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.25       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.30       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.35       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.40       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.45       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.50       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.55       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.60       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.65       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.70       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.75       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
0.80       0.8873   0.8984   0.7778   0.9959   0.9968   0.8177  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8873, F1=0.8984, Normal Recall=0.7778, Normal Precision=0.9959, Attack Recall=0.9968, Attack Precision=0.8177

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
0.15       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332   <--
0.20       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.25       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.30       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.35       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.40       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.45       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.50       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.55       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.60       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.65       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.70       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.75       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.80       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8002, F1=0.4994, Normal Recall=0.7783, Normal Precision=0.9995, Attack Recall=0.9968, Attack Precision=0.3332

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
0.15       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300   <--
0.20       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.25       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.30       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.35       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.40       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.45       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.50       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.55       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.60       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.65       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.70       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.75       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.80       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8226, F1=0.6920, Normal Recall=0.7790, Normal Precision=0.9989, Attack Recall=0.9967, Attack Precision=0.5300

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
0.15       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590   <--
0.20       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.25       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.30       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.35       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.40       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.45       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.50       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.55       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.60       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.65       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.70       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.75       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.80       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8443, F1=0.7934, Normal Recall=0.7790, Normal Precision=0.9982, Attack Recall=0.9967, Attack Precision=0.6590

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
0.15       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498   <--
0.20       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.25       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.30       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.35       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.40       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.45       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.50       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.55       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.60       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.65       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.70       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.75       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.80       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8656, F1=0.8558, Normal Recall=0.7783, Normal Precision=0.9971, Attack Recall=0.9967, Attack Precision=0.7498

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
0.15       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165   <--
0.20       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.25       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.30       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.35       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.40       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.45       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.50       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.55       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.60       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.65       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.70       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.75       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.80       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8863, F1=0.8976, Normal Recall=0.7760, Normal Precision=0.9957, Attack Recall=0.9967, Attack Precision=0.8165

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
0.15       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332   <--
0.20       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.25       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.30       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.35       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.40       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.45       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.50       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.55       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.60       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.65       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.70       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.75       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
0.80       0.8002   0.4994   0.7783   0.9995   0.9968   0.3332  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8002, F1=0.4994, Normal Recall=0.7783, Normal Precision=0.9995, Attack Recall=0.9968, Attack Precision=0.3332

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
0.15       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300   <--
0.20       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.25       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.30       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.35       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.40       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.45       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.50       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.55       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.60       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.65       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.70       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.75       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
0.80       0.8226   0.6920   0.7790   0.9989   0.9967   0.5300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8226, F1=0.6920, Normal Recall=0.7790, Normal Precision=0.9989, Attack Recall=0.9967, Attack Precision=0.5300

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
0.15       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590   <--
0.20       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.25       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.30       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.35       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.40       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.45       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.50       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.55       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.60       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.65       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.70       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.75       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
0.80       0.8443   0.7934   0.7790   0.9982   0.9967   0.6590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8443, F1=0.7934, Normal Recall=0.7790, Normal Precision=0.9982, Attack Recall=0.9967, Attack Precision=0.6590

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
0.15       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498   <--
0.20       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.25       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.30       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.35       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.40       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.45       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.50       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.55       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.60       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.65       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.70       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.75       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
0.80       0.8656   0.8558   0.7783   0.9971   0.9967   0.7498  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8656, F1=0.8558, Normal Recall=0.7783, Normal Precision=0.9971, Attack Recall=0.9967, Attack Precision=0.7498

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
0.15       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165   <--
0.20       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.25       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.30       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.35       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.40       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.45       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.50       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.55       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.60       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.65       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.70       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.75       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
0.80       0.8863   0.8976   0.7760   0.9957   0.9967   0.8165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8863, F1=0.8976, Normal Recall=0.7760, Normal Precision=0.9957, Attack Recall=0.9967, Attack Precision=0.8165

```

