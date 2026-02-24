# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-17 10:00:54 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1339 | 0.2206 | 0.3070 | 0.3943 | 0.4808 | 0.5664 | 0.6537 | 0.7416 | 0.8264 | 0.9141 | 0.9998 |
| QAT+Prune only | 0.4133 | 0.4665 | 0.5177 | 0.5697 | 0.6230 | 0.6725 | 0.7247 | 0.7769 | 0.8279 | 0.8810 | 0.9318 |
| QAT+PTQ | 0.4133 | 0.4664 | 0.5176 | 0.5697 | 0.6230 | 0.6724 | 0.7246 | 0.7769 | 0.8278 | 0.8809 | 0.9317 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4133 | 0.4664 | 0.5176 | 0.5697 | 0.6230 | 0.6724 | 0.7246 | 0.7769 | 0.8278 | 0.8809 | 0.9317 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2042 | 0.3659 | 0.4976 | 0.6064 | 0.6975 | 0.7760 | 0.8442 | 0.9021 | 0.9544 | 0.9999 |
| QAT+Prune only | 0.0000 | 0.2592 | 0.4359 | 0.5651 | 0.6641 | 0.7399 | 0.8024 | 0.8540 | 0.8965 | 0.9338 | 0.9647 |
| QAT+PTQ | 0.0000 | 0.2592 | 0.4358 | 0.5651 | 0.6641 | 0.7399 | 0.8024 | 0.8539 | 0.8965 | 0.9337 | 0.9647 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2592 | 0.4358 | 0.5651 | 0.6641 | 0.7399 | 0.8024 | 0.8539 | 0.8965 | 0.9337 | 0.9647 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1339 | 0.1340 | 0.1338 | 0.1347 | 0.1348 | 0.1331 | 0.1346 | 0.1391 | 0.1329 | 0.1423 | 0.0000 |
| QAT+Prune only | 0.4133 | 0.4146 | 0.4141 | 0.4146 | 0.4172 | 0.4131 | 0.4140 | 0.4155 | 0.4125 | 0.4238 | 0.0000 |
| QAT+PTQ | 0.4133 | 0.4145 | 0.4140 | 0.4145 | 0.4171 | 0.4131 | 0.4139 | 0.4155 | 0.4120 | 0.4231 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4133 | 0.4145 | 0.4140 | 0.4145 | 0.4171 | 0.4131 | 0.4139 | 0.4155 | 0.4120 | 0.4231 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1339 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 1.0000 |
| 90 | 10 | 299,940 | 0.2206 | 0.1137 | 0.9998 | 0.2042 | 0.1340 | 0.9999 |
| 80 | 20 | 291,350 | 0.3070 | 0.2239 | 0.9998 | 0.3659 | 0.1338 | 0.9996 |
| 70 | 30 | 194,230 | 0.3943 | 0.3312 | 0.9998 | 0.4976 | 0.1347 | 0.9994 |
| 60 | 40 | 145,675 | 0.4808 | 0.4352 | 0.9998 | 0.6064 | 0.1348 | 0.9991 |
| 50 | 50 | 116,540 | 0.5664 | 0.5356 | 0.9998 | 0.6975 | 0.1331 | 0.9986 |
| 40 | 60 | 97,115 | 0.6537 | 0.6341 | 0.9998 | 0.7760 | 0.1346 | 0.9979 |
| 30 | 70 | 83,240 | 0.7416 | 0.7304 | 0.9998 | 0.8442 | 0.1391 | 0.9968 |
| 20 | 80 | 72,835 | 0.8264 | 0.8218 | 0.9998 | 0.9021 | 0.1329 | 0.9944 |
| 10 | 90 | 64,740 | 0.9141 | 0.9130 | 0.9998 | 0.9544 | 0.1423 | 0.9882 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4133 | 0.0000 | 0.0000 | 0.0000 | 0.4133 | 1.0000 |
| 90 | 10 | 299,940 | 0.4665 | 0.1505 | 0.9333 | 0.2592 | 0.4146 | 0.9824 |
| 80 | 20 | 291,350 | 0.5177 | 0.2845 | 0.9318 | 0.4359 | 0.4141 | 0.9605 |
| 70 | 30 | 194,230 | 0.5697 | 0.4055 | 0.9318 | 0.5651 | 0.4146 | 0.9341 |
| 60 | 40 | 145,675 | 0.6230 | 0.5159 | 0.9318 | 0.6641 | 0.4172 | 0.9017 |
| 50 | 50 | 116,540 | 0.6725 | 0.6136 | 0.9318 | 0.7399 | 0.4131 | 0.8583 |
| 40 | 60 | 97,115 | 0.7247 | 0.7046 | 0.9318 | 0.8024 | 0.4140 | 0.8019 |
| 30 | 70 | 83,240 | 0.7769 | 0.7881 | 0.9318 | 0.8540 | 0.4155 | 0.7231 |
| 20 | 80 | 72,835 | 0.8279 | 0.8638 | 0.9318 | 0.8965 | 0.4125 | 0.6019 |
| 10 | 90 | 64,740 | 0.8810 | 0.9357 | 0.9318 | 0.9338 | 0.4238 | 0.4085 |
| 0 | 100 | 58,270 | 0.9318 | 1.0000 | 0.9318 | 0.9647 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4133 | 0.0000 | 0.0000 | 0.0000 | 0.4133 | 1.0000 |
| 90 | 10 | 299,940 | 0.4664 | 0.1505 | 0.9333 | 0.2592 | 0.4145 | 0.9824 |
| 80 | 20 | 291,350 | 0.5176 | 0.2845 | 0.9317 | 0.4358 | 0.4140 | 0.9604 |
| 70 | 30 | 194,230 | 0.5697 | 0.4055 | 0.9317 | 0.5651 | 0.4145 | 0.9341 |
| 60 | 40 | 145,675 | 0.6230 | 0.5159 | 0.9317 | 0.6641 | 0.4171 | 0.9016 |
| 50 | 50 | 116,540 | 0.6724 | 0.6135 | 0.9317 | 0.7399 | 0.4131 | 0.8582 |
| 40 | 60 | 97,115 | 0.7246 | 0.7046 | 0.9317 | 0.8024 | 0.4139 | 0.8017 |
| 30 | 70 | 83,240 | 0.7769 | 0.7881 | 0.9317 | 0.8539 | 0.4155 | 0.7229 |
| 20 | 80 | 72,835 | 0.8278 | 0.8637 | 0.9317 | 0.8965 | 0.4120 | 0.6015 |
| 10 | 90 | 64,740 | 0.8809 | 0.9356 | 0.9317 | 0.9337 | 0.4231 | 0.4078 |
| 0 | 100 | 58,270 | 0.9317 | 1.0000 | 0.9317 | 0.9647 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4133 | 0.0000 | 0.0000 | 0.0000 | 0.4133 | 1.0000 |
| 90 | 10 | 299,940 | 0.4664 | 0.1505 | 0.9333 | 0.2592 | 0.4145 | 0.9824 |
| 80 | 20 | 291,350 | 0.5176 | 0.2845 | 0.9317 | 0.4358 | 0.4140 | 0.9604 |
| 70 | 30 | 194,230 | 0.5697 | 0.4055 | 0.9317 | 0.5651 | 0.4145 | 0.9341 |
| 60 | 40 | 145,675 | 0.6230 | 0.5159 | 0.9317 | 0.6641 | 0.4171 | 0.9016 |
| 50 | 50 | 116,540 | 0.6724 | 0.6135 | 0.9317 | 0.7399 | 0.4131 | 0.8582 |
| 40 | 60 | 97,115 | 0.7246 | 0.7046 | 0.9317 | 0.8024 | 0.4139 | 0.8017 |
| 30 | 70 | 83,240 | 0.7769 | 0.7881 | 0.9317 | 0.8539 | 0.4155 | 0.7229 |
| 20 | 80 | 72,835 | 0.8278 | 0.8637 | 0.9317 | 0.8965 | 0.4120 | 0.6015 |
| 10 | 90 | 64,740 | 0.8809 | 0.9356 | 0.9317 | 0.9337 | 0.4231 | 0.4078 |
| 0 | 100 | 58,270 | 0.9317 | 1.0000 | 0.9317 | 0.9647 | 0.0000 | 0.0000 |


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
0.15       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137   <--
0.20       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.25       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.30       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.35       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.40       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.45       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.50       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.55       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.60       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.65       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.70       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.75       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
0.80       0.2206   0.2042   0.1340   0.9998   0.9998   0.1137  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2206, F1=0.2042, Normal Recall=0.1340, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.1137

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
0.15       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239   <--
0.20       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.25       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.30       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.35       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.40       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.45       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.50       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.55       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.60       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.65       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.70       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.75       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
0.80       0.3069   0.3659   0.1337   0.9996   0.9998   0.2239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3069, F1=0.3659, Normal Recall=0.1337, Normal Precision=0.9996, Attack Recall=0.9998, Attack Precision=0.2239

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
0.15       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310   <--
0.20       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.25       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.30       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.35       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.40       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.45       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.50       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.55       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.60       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.65       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.70       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.75       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
0.80       0.3938   0.4974   0.1341   0.9994   0.9998   0.3310  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3938, F1=0.4974, Normal Recall=0.1341, Normal Precision=0.9994, Attack Recall=0.9998, Attack Precision=0.3310

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
0.15       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351   <--
0.20       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.25       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.30       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.35       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.40       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.45       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.50       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.55       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.60       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.65       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.70       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.75       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
0.80       0.4807   0.6063   0.1346   0.9991   0.9998   0.4351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4807, F1=0.6063, Normal Recall=0.1346, Normal Precision=0.9991, Attack Recall=0.9998, Attack Precision=0.4351

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
0.15       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358   <--
0.20       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.25       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.30       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.35       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.40       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.45       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.50       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.55       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.60       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.65       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.70       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.75       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
0.80       0.5669   0.6977   0.1339   0.9986   0.9998   0.5358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5669, F1=0.6977, Normal Recall=0.1339, Normal Precision=0.9986, Attack Recall=0.9998, Attack Precision=0.5358

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
0.15       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503   <--
0.20       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.25       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.30       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.35       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.40       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.45       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.50       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.55       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.60       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.65       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.70       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.75       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
0.80       0.4663   0.2588   0.4146   0.9820   0.9317   0.1503  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4663, F1=0.2588, Normal Recall=0.4146, Normal Precision=0.9820, Attack Recall=0.9317, Attack Precision=0.1503

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
0.15       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844   <--
0.20       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.25       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.30       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.35       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.40       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.45       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.50       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.55       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.60       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.65       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.70       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.75       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
0.80       0.5175   0.4358   0.4139   0.9604   0.9318   0.2844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5175, F1=0.4358, Normal Recall=0.4139, Normal Precision=0.9604, Attack Recall=0.9318, Attack Precision=0.2844

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
0.15       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053   <--
0.20       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.25       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.30       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.35       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.40       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.45       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.50       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.55       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.60       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.65       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.70       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.75       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
0.80       0.5694   0.5649   0.4140   0.9341   0.9318   0.4053  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5694, F1=0.5649, Normal Recall=0.4140, Normal Precision=0.9341, Attack Recall=0.9318, Attack Precision=0.4053

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
0.15       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143   <--
0.20       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.25       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.30       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.35       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.40       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.45       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.50       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.55       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.60       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.65       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.70       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.75       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
0.80       0.6208   0.6628   0.4134   0.9009   0.9318   0.5143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6208, F1=0.6628, Normal Recall=0.4134, Normal Precision=0.9009, Attack Recall=0.9318, Attack Precision=0.5143

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
0.15       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136   <--
0.20       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.25       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.30       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.35       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.40       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.45       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.50       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.55       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.60       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.65       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.70       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.75       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
0.80       0.6725   0.7399   0.4132   0.8583   0.9318   0.6136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6725, F1=0.7399, Normal Recall=0.4132, Normal Precision=0.8583, Attack Recall=0.9318, Attack Precision=0.6136

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
0.15       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502   <--
0.20       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.25       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.30       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.35       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.40       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.45       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.50       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.55       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.60       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.65       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.70       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.75       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.80       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4662, F1=0.2588, Normal Recall=0.4145, Normal Precision=0.9820, Attack Recall=0.9317, Attack Precision=0.1502

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
0.15       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844   <--
0.20       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.25       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.30       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.35       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.40       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.45       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.50       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.55       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.60       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.65       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.70       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.75       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.80       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5174, F1=0.4357, Normal Recall=0.4138, Normal Precision=0.9604, Attack Recall=0.9317, Attack Precision=0.2844

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
0.15       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053   <--
0.20       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.25       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.30       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.35       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.40       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.45       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.50       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.55       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.60       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.65       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.70       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.75       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.80       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5693, F1=0.5649, Normal Recall=0.4140, Normal Precision=0.9340, Attack Recall=0.9317, Attack Precision=0.4053

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
0.15       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144   <--
0.20       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.25       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.30       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.35       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.40       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.45       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.50       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.55       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.60       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.65       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.70       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.75       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.80       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6208, F1=0.6628, Normal Recall=0.4135, Normal Precision=0.9009, Attack Recall=0.9317, Attack Precision=0.5144

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
0.15       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136   <--
0.20       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.25       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.30       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.35       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.40       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.45       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.50       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.55       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.60       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.65       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.70       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.75       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.80       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6726, F1=0.7400, Normal Recall=0.4134, Normal Precision=0.8583, Attack Recall=0.9317, Attack Precision=0.6136

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
0.15       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502   <--
0.20       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.25       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.30       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.35       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.40       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.45       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.50       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.55       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.60       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.65       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.70       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.75       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
0.80       0.4662   0.2588   0.4145   0.9820   0.9317   0.1502  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4662, F1=0.2588, Normal Recall=0.4145, Normal Precision=0.9820, Attack Recall=0.9317, Attack Precision=0.1502

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
0.15       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844   <--
0.20       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.25       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.30       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.35       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.40       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.45       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.50       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.55       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.60       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.65       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.70       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.75       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
0.80       0.5174   0.4357   0.4138   0.9604   0.9317   0.2844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5174, F1=0.4357, Normal Recall=0.4138, Normal Precision=0.9604, Attack Recall=0.9317, Attack Precision=0.2844

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
0.15       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053   <--
0.20       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.25       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.30       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.35       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.40       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.45       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.50       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.55       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.60       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.65       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.70       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.75       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
0.80       0.5693   0.5649   0.4140   0.9340   0.9317   0.4053  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5693, F1=0.5649, Normal Recall=0.4140, Normal Precision=0.9340, Attack Recall=0.9317, Attack Precision=0.4053

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
0.15       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144   <--
0.20       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.25       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.30       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.35       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.40       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.45       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.50       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.55       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.60       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.65       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.70       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.75       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
0.80       0.6208   0.6628   0.4135   0.9009   0.9317   0.5144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6208, F1=0.6628, Normal Recall=0.4135, Normal Precision=0.9009, Attack Recall=0.9317, Attack Precision=0.5144

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
0.15       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136   <--
0.20       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.25       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.30       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.35       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.40       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.45       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.50       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.55       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.60       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.65       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.70       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.75       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
0.80       0.6726   0.7400   0.4134   0.8583   0.9317   0.6136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6726, F1=0.7400, Normal Recall=0.4134, Normal Precision=0.8583, Attack Recall=0.9317, Attack Precision=0.6136

```

