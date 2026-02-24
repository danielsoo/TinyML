# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-22 01:56:51 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3777 | 0.4396 | 0.5013 | 0.5638 | 0.6264 | 0.6870 | 0.7496 | 0.8111 | 0.8722 | 0.9354 | 0.9966 |
| QAT+Prune only | 0.8012 | 0.8003 | 0.7974 | 0.7953 | 0.7922 | 0.7889 | 0.7868 | 0.7838 | 0.7807 | 0.7778 | 0.7760 |
| QAT+PTQ | 0.8055 | 0.8064 | 0.8053 | 0.8051 | 0.8040 | 0.8021 | 0.8026 | 0.8015 | 0.8002 | 0.7994 | 0.7994 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8055 | 0.8064 | 0.8053 | 0.8051 | 0.8040 | 0.8021 | 0.8026 | 0.8015 | 0.8002 | 0.7994 | 0.7994 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2623 | 0.4443 | 0.5782 | 0.6809 | 0.7610 | 0.8269 | 0.8808 | 0.9258 | 0.9652 | 0.9983 |
| QAT+Prune only | 0.0000 | 0.4371 | 0.6050 | 0.6946 | 0.7492 | 0.7861 | 0.8137 | 0.8340 | 0.8499 | 0.8628 | 0.8739 |
| QAT+PTQ | 0.0000 | 0.4523 | 0.6215 | 0.7111 | 0.7654 | 0.8016 | 0.8293 | 0.8494 | 0.8649 | 0.8776 | 0.8885 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4523 | 0.6215 | 0.7111 | 0.7654 | 0.8016 | 0.8293 | 0.8494 | 0.8649 | 0.8776 | 0.8885 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3777 | 0.3777 | 0.3775 | 0.3782 | 0.3795 | 0.3773 | 0.3791 | 0.3783 | 0.3743 | 0.3840 | 0.0000 |
| QAT+Prune only | 0.8012 | 0.8031 | 0.8027 | 0.8036 | 0.8029 | 0.8017 | 0.8029 | 0.8021 | 0.7996 | 0.7943 | 0.0000 |
| QAT+PTQ | 0.8055 | 0.8072 | 0.8067 | 0.8076 | 0.8071 | 0.8049 | 0.8074 | 0.8065 | 0.8033 | 0.7990 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8055 | 0.8072 | 0.8067 | 0.8076 | 0.8071 | 0.8049 | 0.8074 | 0.8065 | 0.8033 | 0.7990 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3777 | 0.0000 | 0.0000 | 0.0000 | 0.3777 | 1.0000 |
| 90 | 10 | 299,940 | 0.4396 | 0.1511 | 0.9966 | 0.2623 | 0.3777 | 0.9990 |
| 80 | 20 | 291,350 | 0.5013 | 0.2858 | 0.9966 | 0.4443 | 0.3775 | 0.9978 |
| 70 | 30 | 194,230 | 0.5638 | 0.4072 | 0.9966 | 0.5782 | 0.3782 | 0.9962 |
| 60 | 40 | 145,675 | 0.6264 | 0.5171 | 0.9966 | 0.6809 | 0.3795 | 0.9941 |
| 50 | 50 | 116,540 | 0.6870 | 0.6155 | 0.9966 | 0.7610 | 0.3773 | 0.9911 |
| 40 | 60 | 97,115 | 0.7496 | 0.7065 | 0.9966 | 0.8269 | 0.3791 | 0.9868 |
| 30 | 70 | 83,240 | 0.8111 | 0.7890 | 0.9966 | 0.8808 | 0.3783 | 0.9796 |
| 20 | 80 | 72,835 | 0.8722 | 0.8643 | 0.9966 | 0.9258 | 0.3743 | 0.9651 |
| 10 | 90 | 64,740 | 0.9354 | 0.9357 | 0.9966 | 0.9652 | 0.3840 | 0.9266 |
| 0 | 100 | 58,270 | 0.9966 | 1.0000 | 0.9966 | 0.9983 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8012 | 0.0000 | 0.0000 | 0.0000 | 0.8012 | 1.0000 |
| 90 | 10 | 299,940 | 0.8003 | 0.3043 | 0.7752 | 0.4371 | 0.8031 | 0.9698 |
| 80 | 20 | 291,350 | 0.7974 | 0.4958 | 0.7760 | 0.6050 | 0.8027 | 0.9348 |
| 70 | 30 | 194,230 | 0.7953 | 0.6287 | 0.7760 | 0.6946 | 0.8036 | 0.8933 |
| 60 | 40 | 145,675 | 0.7922 | 0.7241 | 0.7760 | 0.7492 | 0.8029 | 0.8432 |
| 50 | 50 | 116,540 | 0.7889 | 0.7965 | 0.7760 | 0.7861 | 0.8017 | 0.7816 |
| 40 | 60 | 97,115 | 0.7868 | 0.8552 | 0.7760 | 0.8137 | 0.8029 | 0.7050 |
| 30 | 70 | 83,240 | 0.7838 | 0.9015 | 0.7760 | 0.8340 | 0.8021 | 0.6055 |
| 20 | 80 | 72,835 | 0.7807 | 0.9394 | 0.7760 | 0.8499 | 0.7996 | 0.4716 |
| 10 | 90 | 64,740 | 0.7778 | 0.9714 | 0.7760 | 0.8628 | 0.7943 | 0.2827 |
| 0 | 100 | 58,270 | 0.7760 | 1.0000 | 0.7760 | 0.8739 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8055 | 0.0000 | 0.0000 | 0.0000 | 0.8055 | 1.0000 |
| 90 | 10 | 299,940 | 0.8064 | 0.3154 | 0.7993 | 0.4523 | 0.8072 | 0.9731 |
| 80 | 20 | 291,350 | 0.8053 | 0.5084 | 0.7994 | 0.6215 | 0.8067 | 0.9415 |
| 70 | 30 | 194,230 | 0.8051 | 0.6404 | 0.7994 | 0.7111 | 0.8076 | 0.9038 |
| 60 | 40 | 145,675 | 0.8040 | 0.7342 | 0.7994 | 0.7654 | 0.8071 | 0.8578 |
| 50 | 50 | 116,540 | 0.8021 | 0.8038 | 0.7994 | 0.8016 | 0.8049 | 0.8005 |
| 40 | 60 | 97,115 | 0.8026 | 0.8616 | 0.7994 | 0.8293 | 0.8074 | 0.7285 |
| 30 | 70 | 83,240 | 0.8015 | 0.9060 | 0.7994 | 0.8494 | 0.8065 | 0.6327 |
| 20 | 80 | 72,835 | 0.8002 | 0.9421 | 0.7994 | 0.8649 | 0.8033 | 0.5003 |
| 10 | 90 | 64,740 | 0.7994 | 0.9728 | 0.7994 | 0.8776 | 0.7990 | 0.3068 |
| 0 | 100 | 58,270 | 0.7994 | 1.0000 | 0.7994 | 0.8885 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8055 | 0.0000 | 0.0000 | 0.0000 | 0.8055 | 1.0000 |
| 90 | 10 | 299,940 | 0.8064 | 0.3154 | 0.7993 | 0.4523 | 0.8072 | 0.9731 |
| 80 | 20 | 291,350 | 0.8053 | 0.5084 | 0.7994 | 0.6215 | 0.8067 | 0.9415 |
| 70 | 30 | 194,230 | 0.8051 | 0.6404 | 0.7994 | 0.7111 | 0.8076 | 0.9038 |
| 60 | 40 | 145,675 | 0.8040 | 0.7342 | 0.7994 | 0.7654 | 0.8071 | 0.8578 |
| 50 | 50 | 116,540 | 0.8021 | 0.8038 | 0.7994 | 0.8016 | 0.8049 | 0.8005 |
| 40 | 60 | 97,115 | 0.8026 | 0.8616 | 0.7994 | 0.8293 | 0.8074 | 0.7285 |
| 30 | 70 | 83,240 | 0.8015 | 0.9060 | 0.7994 | 0.8494 | 0.8065 | 0.6327 |
| 20 | 80 | 72,835 | 0.8002 | 0.9421 | 0.7994 | 0.8649 | 0.8033 | 0.5003 |
| 10 | 90 | 64,740 | 0.7994 | 0.9728 | 0.7994 | 0.8776 | 0.7990 | 0.3068 |
| 0 | 100 | 58,270 | 0.7994 | 1.0000 | 0.7994 | 0.8885 | 0.0000 | 0.0000 |


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
0.15       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511   <--
0.20       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.25       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.30       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.35       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.40       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.45       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.50       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.55       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.60       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.65       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.70       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.75       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
0.80       0.4396   0.2624   0.3777   0.9991   0.9969   0.1511  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4396, F1=0.2624, Normal Recall=0.3777, Normal Precision=0.9991, Attack Recall=0.9969, Attack Precision=0.1511

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
0.15       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859   <--
0.20       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.25       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.30       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.35       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.40       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.45       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.50       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.55       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.60       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.65       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.70       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.75       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
0.80       0.5015   0.4443   0.3777   0.9978   0.9966   0.2859  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5015, F1=0.4443, Normal Recall=0.3777, Normal Precision=0.9978, Attack Recall=0.9966, Attack Precision=0.2859

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
0.15       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073   <--
0.20       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.25       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.30       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.35       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.40       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.45       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.50       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.55       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.60       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.65       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.70       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.75       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
0.80       0.5639   0.5783   0.3784   0.9962   0.9966   0.4073  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5639, F1=0.5783, Normal Recall=0.3784, Normal Precision=0.9962, Attack Recall=0.9966, Attack Precision=0.4073

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
0.15       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161   <--
0.20       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.25       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.30       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.35       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.40       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.45       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.50       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.55       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.60       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.65       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.70       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.75       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
0.80       0.6249   0.6801   0.3771   0.9941   0.9966   0.5161  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6249, F1=0.6801, Normal Recall=0.3771, Normal Precision=0.9941, Attack Recall=0.9966, Attack Precision=0.5161

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
0.15       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151   <--
0.20       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.25       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.30       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.35       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.40       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.45       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.50       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.55       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.60       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.65       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.70       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.75       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
0.80       0.6864   0.7607   0.3762   0.9911   0.9966   0.6151  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6864, F1=0.7607, Normal Recall=0.3762, Normal Precision=0.9911, Attack Recall=0.9966, Attack Precision=0.6151

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
0.15       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056   <--
0.20       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.25       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.30       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.35       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.40       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.45       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.50       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.55       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.60       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.65       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.70       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.75       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
0.80       0.8008   0.4391   0.8031   0.9704   0.7798   0.3056  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8008, F1=0.4391, Normal Recall=0.8031, Normal Precision=0.9704, Attack Recall=0.7798, Attack Precision=0.3056

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
0.15       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966   <--
0.20       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.25       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.30       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.35       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.40       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.45       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.50       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.55       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.60       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.65       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.70       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.75       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
0.80       0.7979   0.6056   0.8033   0.9348   0.7760   0.4966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7979, F1=0.6056, Normal Recall=0.8033, Normal Precision=0.9348, Attack Recall=0.7760, Attack Precision=0.4966

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
0.15       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260   <--
0.20       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.25       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.30       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.35       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.40       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.45       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.50       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.55       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.60       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.65       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.70       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.75       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
0.80       0.7937   0.6930   0.8013   0.8930   0.7760   0.6260  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7937, F1=0.6930, Normal Recall=0.8013, Normal Precision=0.8930, Attack Recall=0.7760, Attack Precision=0.6260

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
0.15       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224   <--
0.20       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.25       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.30       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.35       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.40       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.45       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.50       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.55       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.60       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.65       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.70       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.75       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
0.80       0.7911   0.7482   0.8012   0.8429   0.7760   0.7224  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7911, F1=0.7482, Normal Recall=0.8012, Normal Precision=0.8429, Attack Recall=0.7760, Attack Precision=0.7224

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
0.15       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963   <--
0.20       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.25       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.30       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.35       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.40       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.45       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.50       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.55       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.60       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.65       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.70       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.75       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
0.80       0.7887   0.7860   0.8015   0.7816   0.7760   0.7963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7887, F1=0.7860, Normal Recall=0.8015, Normal Precision=0.7816, Attack Recall=0.7760, Attack Precision=0.7963

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
0.15       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163   <--
0.20       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.25       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.30       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.35       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.40       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.45       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.50       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.55       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.60       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.65       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.70       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.75       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.80       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8067, F1=0.4537, Normal Recall=0.8072, Normal Precision=0.9735, Attack Recall=0.8025, Attack Precision=0.3163

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
0.15       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094   <--
0.20       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.25       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.30       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.35       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.40       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.45       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.50       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.55       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.60       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.65       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.70       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.75       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.80       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8059, F1=0.6223, Normal Recall=0.8075, Normal Precision=0.9415, Attack Recall=0.7994, Attack Precision=0.5094

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
0.15       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380   <--
0.20       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.25       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.30       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.35       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.40       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.45       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.50       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.55       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.60       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.65       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.70       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.75       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.80       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8038, F1=0.7096, Normal Recall=0.8056, Normal Precision=0.9036, Attack Recall=0.7994, Attack Precision=0.6380

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
0.15       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326   <--
0.20       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.25       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.30       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.35       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.40       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.45       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.50       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.55       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.60       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.65       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.70       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.75       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.80       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8030, F1=0.7645, Normal Recall=0.8055, Normal Precision=0.8576, Attack Recall=0.7994, Attack Precision=0.7326

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
0.15       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046   <--
0.20       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.25       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.30       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.35       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.40       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.45       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.50       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.55       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.60       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.65       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.70       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.75       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.80       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8026, F1=0.8020, Normal Recall=0.8058, Normal Precision=0.8007, Attack Recall=0.7994, Attack Precision=0.8046

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
0.15       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163   <--
0.20       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.25       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.30       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.35       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.40       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.45       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.50       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.55       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.60       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.65       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.70       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.75       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
0.80       0.8067   0.4537   0.8072   0.9735   0.8025   0.3163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8067, F1=0.4537, Normal Recall=0.8072, Normal Precision=0.9735, Attack Recall=0.8025, Attack Precision=0.3163

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
0.15       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094   <--
0.20       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.25       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.30       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.35       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.40       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.45       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.50       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.55       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.60       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.65       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.70       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.75       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
0.80       0.8059   0.6223   0.8075   0.9415   0.7994   0.5094  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8059, F1=0.6223, Normal Recall=0.8075, Normal Precision=0.9415, Attack Recall=0.7994, Attack Precision=0.5094

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
0.15       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380   <--
0.20       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.25       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.30       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.35       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.40       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.45       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.50       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.55       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.60       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.65       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.70       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.75       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
0.80       0.8038   0.7096   0.8056   0.9036   0.7994   0.6380  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8038, F1=0.7096, Normal Recall=0.8056, Normal Precision=0.9036, Attack Recall=0.7994, Attack Precision=0.6380

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
0.15       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326   <--
0.20       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.25       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.30       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.35       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.40       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.45       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.50       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.55       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.60       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.65       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.70       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.75       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
0.80       0.8030   0.7645   0.8055   0.8576   0.7994   0.7326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8030, F1=0.7645, Normal Recall=0.8055, Normal Precision=0.8576, Attack Recall=0.7994, Attack Precision=0.7326

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
0.15       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046   <--
0.20       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.25       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.30       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.35       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.40       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.45       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.50       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.55       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.60       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.65       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.70       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.75       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
0.80       0.8026   0.8020   0.8058   0.8007   0.7994   0.8046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8026, F1=0.8020, Normal Recall=0.8058, Normal Precision=0.8007, Attack Recall=0.7994, Attack Precision=0.8046

```

