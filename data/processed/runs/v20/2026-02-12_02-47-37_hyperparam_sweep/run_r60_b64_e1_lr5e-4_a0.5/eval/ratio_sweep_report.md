# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-12 08:06:48 |

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
| Original (TFLite) | 0.7565 | 0.7768 | 0.7963 | 0.8174 | 0.8370 | 0.8577 | 0.8771 | 0.8967 | 0.9172 | 0.9371 | 0.9575 |
| QAT+Prune only | 0.8071 | 0.8222 | 0.8373 | 0.8539 | 0.8691 | 0.8839 | 0.9001 | 0.9153 | 0.9308 | 0.9470 | 0.9620 |
| QAT+PTQ | 0.8068 | 0.8221 | 0.8374 | 0.8544 | 0.8697 | 0.8847 | 0.9012 | 0.9165 | 0.9321 | 0.9486 | 0.9640 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8068 | 0.8221 | 0.8374 | 0.8544 | 0.8697 | 0.8847 | 0.9012 | 0.9165 | 0.9321 | 0.9486 | 0.9640 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4622 | 0.6528 | 0.7589 | 0.8245 | 0.8706 | 0.9034 | 0.9285 | 0.9487 | 0.9648 | 0.9783 |
| QAT+Prune only | 0.0000 | 0.5196 | 0.7029 | 0.7980 | 0.8547 | 0.8924 | 0.9204 | 0.9408 | 0.9570 | 0.9703 | 0.9807 |
| QAT+PTQ | 0.0000 | 0.5201 | 0.7034 | 0.7989 | 0.8555 | 0.8932 | 0.9213 | 0.9417 | 0.9578 | 0.9712 | 0.9817 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5201 | 0.7034 | 0.7989 | 0.8555 | 0.8932 | 0.9213 | 0.9417 | 0.9578 | 0.9712 | 0.9817 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7565 | 0.7566 | 0.7560 | 0.7574 | 0.7566 | 0.7578 | 0.7566 | 0.7550 | 0.7562 | 0.7541 | 0.0000 |
| QAT+Prune only | 0.8071 | 0.8066 | 0.8061 | 0.8075 | 0.8072 | 0.8059 | 0.8073 | 0.8063 | 0.8057 | 0.8112 | 0.0000 |
| QAT+PTQ | 0.8068 | 0.8063 | 0.8058 | 0.8074 | 0.8069 | 0.8055 | 0.8071 | 0.8057 | 0.8044 | 0.8102 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8068 | 0.8063 | 0.8058 | 0.8074 | 0.8069 | 0.8055 | 0.8071 | 0.8057 | 0.8044 | 0.8102 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7565 | 0.0000 | 0.0000 | 0.0000 | 0.7565 | 1.0000 |
| 90 | 10 | 299,940 | 0.7768 | 0.3045 | 0.9590 | 0.4622 | 0.7566 | 0.9940 |
| 80 | 20 | 291,350 | 0.7963 | 0.4952 | 0.9575 | 0.6528 | 0.7560 | 0.9861 |
| 70 | 30 | 194,230 | 0.8174 | 0.6285 | 0.9575 | 0.7589 | 0.7574 | 0.9765 |
| 60 | 40 | 145,675 | 0.8370 | 0.7240 | 0.9575 | 0.8245 | 0.7566 | 0.9639 |
| 50 | 50 | 116,540 | 0.8577 | 0.7981 | 0.9575 | 0.8706 | 0.7578 | 0.9469 |
| 40 | 60 | 97,115 | 0.8771 | 0.8551 | 0.9575 | 0.9034 | 0.7566 | 0.9222 |
| 30 | 70 | 83,240 | 0.8967 | 0.9012 | 0.9575 | 0.9285 | 0.7550 | 0.8838 |
| 20 | 80 | 72,835 | 0.9172 | 0.9401 | 0.9575 | 0.9487 | 0.7562 | 0.8163 |
| 10 | 90 | 64,740 | 0.9371 | 0.9723 | 0.9575 | 0.9648 | 0.7541 | 0.6633 |
| 0 | 100 | 58,270 | 0.9575 | 1.0000 | 0.9575 | 0.9783 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8071 | 0.0000 | 0.0000 | 0.0000 | 0.8071 | 1.0000 |
| 90 | 10 | 299,940 | 0.8222 | 0.3560 | 0.9619 | 0.5196 | 0.8066 | 0.9948 |
| 80 | 20 | 291,350 | 0.8373 | 0.5537 | 0.9620 | 0.7029 | 0.8061 | 0.9884 |
| 70 | 30 | 194,230 | 0.8539 | 0.6818 | 0.9620 | 0.7980 | 0.8075 | 0.9803 |
| 60 | 40 | 145,675 | 0.8691 | 0.7689 | 0.9620 | 0.8547 | 0.8072 | 0.9696 |
| 50 | 50 | 116,540 | 0.8839 | 0.8321 | 0.9620 | 0.8924 | 0.8059 | 0.9550 |
| 40 | 60 | 97,115 | 0.9001 | 0.8822 | 0.9620 | 0.9204 | 0.8073 | 0.9341 |
| 30 | 70 | 83,240 | 0.9153 | 0.9206 | 0.9620 | 0.9408 | 0.8063 | 0.9010 |
| 20 | 80 | 72,835 | 0.9308 | 0.9519 | 0.9620 | 0.9570 | 0.8057 | 0.8414 |
| 10 | 90 | 64,740 | 0.9470 | 0.9787 | 0.9620 | 0.9703 | 0.8112 | 0.7036 |
| 0 | 100 | 58,270 | 0.9620 | 1.0000 | 0.9620 | 0.9807 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8068 | 0.0000 | 0.0000 | 0.0000 | 0.8068 | 1.0000 |
| 90 | 10 | 299,940 | 0.8221 | 0.3561 | 0.9641 | 0.5201 | 0.8063 | 0.9951 |
| 80 | 20 | 291,350 | 0.8374 | 0.5537 | 0.9640 | 0.7034 | 0.8058 | 0.9889 |
| 70 | 30 | 194,230 | 0.8544 | 0.6821 | 0.9640 | 0.7989 | 0.8074 | 0.9812 |
| 60 | 40 | 145,675 | 0.8697 | 0.7690 | 0.9640 | 0.8555 | 0.8069 | 0.9711 |
| 50 | 50 | 116,540 | 0.8847 | 0.8321 | 0.9640 | 0.8932 | 0.8055 | 0.9572 |
| 40 | 60 | 97,115 | 0.9012 | 0.8823 | 0.9640 | 0.9213 | 0.8071 | 0.9373 |
| 30 | 70 | 83,240 | 0.9165 | 0.9205 | 0.9640 | 0.9417 | 0.8057 | 0.9055 |
| 20 | 80 | 72,835 | 0.9321 | 0.9517 | 0.9640 | 0.9578 | 0.8044 | 0.8481 |
| 10 | 90 | 64,740 | 0.9486 | 0.9786 | 0.9640 | 0.9712 | 0.8102 | 0.7142 |
| 0 | 100 | 58,270 | 0.9640 | 1.0000 | 0.9640 | 0.9817 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8068 | 0.0000 | 0.0000 | 0.0000 | 0.8068 | 1.0000 |
| 90 | 10 | 299,940 | 0.8221 | 0.3561 | 0.9641 | 0.5201 | 0.8063 | 0.9951 |
| 80 | 20 | 291,350 | 0.8374 | 0.5537 | 0.9640 | 0.7034 | 0.8058 | 0.9889 |
| 70 | 30 | 194,230 | 0.8544 | 0.6821 | 0.9640 | 0.7989 | 0.8074 | 0.9812 |
| 60 | 40 | 145,675 | 0.8697 | 0.7690 | 0.9640 | 0.8555 | 0.8069 | 0.9711 |
| 50 | 50 | 116,540 | 0.8847 | 0.8321 | 0.9640 | 0.8932 | 0.8055 | 0.9572 |
| 40 | 60 | 97,115 | 0.9012 | 0.8823 | 0.9640 | 0.9213 | 0.8071 | 0.9373 |
| 30 | 70 | 83,240 | 0.9165 | 0.9205 | 0.9640 | 0.9417 | 0.8057 | 0.9055 |
| 20 | 80 | 72,835 | 0.9321 | 0.9517 | 0.9640 | 0.9578 | 0.8044 | 0.8481 |
| 10 | 90 | 64,740 | 0.9486 | 0.9786 | 0.9640 | 0.9712 | 0.8102 | 0.7142 |
| 0 | 100 | 58,270 | 0.9640 | 1.0000 | 0.9640 | 0.9817 | 0.0000 | 0.0000 |


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
0.15       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044   <--
0.20       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.25       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.30       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.35       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.40       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.45       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.50       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.55       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.60       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.65       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.70       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.75       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
0.80       0.7768   0.4621   0.7566   0.9940   0.9586   0.3044  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7768, F1=0.4621, Normal Recall=0.7566, Normal Precision=0.9940, Attack Recall=0.9586, Attack Precision=0.3044

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
0.15       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965   <--
0.20       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.25       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.30       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.35       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.40       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.45       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.50       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.55       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.60       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.65       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.70       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.75       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
0.80       0.7973   0.6539   0.7572   0.9862   0.9575   0.4965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7973, F1=0.6539, Normal Recall=0.7572, Normal Precision=0.9862, Attack Recall=0.9575, Attack Precision=0.4965

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
0.15       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283   <--
0.20       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.25       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.30       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.35       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.40       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.45       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.50       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.55       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.60       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.65       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.70       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.75       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
0.80       0.8173   0.7587   0.7572   0.9765   0.9575   0.6283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.7587, Normal Recall=0.7572, Normal Precision=0.9765, Attack Recall=0.9575, Attack Precision=0.6283

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
0.15       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238   <--
0.20       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.25       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.30       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.35       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.40       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.45       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.50       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.55       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.60       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.65       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.70       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.75       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
0.80       0.8368   0.8244   0.7564   0.9639   0.9575   0.7238  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8368, F1=0.8244, Normal Recall=0.7564, Normal Precision=0.9639, Attack Recall=0.9575, Attack Precision=0.7238

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
0.15       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973   <--
0.20       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.25       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.30       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.35       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.40       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.45       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.50       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.55       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.60       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.65       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.70       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.75       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
0.80       0.8570   0.8701   0.7566   0.9468   0.9575   0.7973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8570, F1=0.8701, Normal Recall=0.7566, Normal Precision=0.9468, Attack Recall=0.9575, Attack Precision=0.7973

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
0.15       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561   <--
0.20       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.25       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.30       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.35       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.40       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.45       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.50       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.55       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.60       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.65       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.70       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.75       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
0.80       0.8222   0.5198   0.8066   0.9948   0.9624   0.3561  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8222, F1=0.5198, Normal Recall=0.8066, Normal Precision=0.9948, Attack Recall=0.9624, Attack Precision=0.3561

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
0.15       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547   <--
0.20       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.25       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.30       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.35       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.40       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.45       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.50       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.55       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.60       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.65       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.70       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.75       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
0.80       0.8380   0.7037   0.8070   0.9884   0.9620   0.5547  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.7037, Normal Recall=0.8070, Normal Precision=0.9884, Attack Recall=0.9620, Attack Precision=0.5547

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
0.15       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819   <--
0.20       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.25       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.30       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.35       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.40       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.45       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.50       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.55       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.60       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.65       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.70       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.75       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
0.80       0.8539   0.7981   0.8076   0.9803   0.9620   0.6819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8539, F1=0.7981, Normal Recall=0.8076, Normal Precision=0.9803, Attack Recall=0.9620, Attack Precision=0.6819

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
0.15       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686   <--
0.20       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.25       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.30       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.35       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.40       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.45       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.50       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.55       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.60       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.65       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.70       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.75       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
0.80       0.8690   0.8545   0.8069   0.9696   0.9620   0.7686  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8690, F1=0.8545, Normal Recall=0.8069, Normal Precision=0.9696, Attack Recall=0.9620, Attack Precision=0.7686

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
0.15       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316   <--
0.20       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.25       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.30       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.35       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.40       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.45       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.50       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.55       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.60       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.65       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.70       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.75       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
0.80       0.8836   0.8921   0.8052   0.9550   0.9620   0.8316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8836, F1=0.8921, Normal Recall=0.8052, Normal Precision=0.9550, Attack Recall=0.9620, Attack Precision=0.8316

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
0.15       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562   <--
0.20       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.25       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.30       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.35       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.40       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.45       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.50       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.55       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.60       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.65       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.70       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.75       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.80       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8221, F1=0.5203, Normal Recall=0.8063, Normal Precision=0.9951, Attack Recall=0.9645, Attack Precision=0.3562

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
0.15       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547   <--
0.20       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.25       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.30       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.35       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.40       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.45       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.50       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.55       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.60       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.65       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.70       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.75       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.80       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.7042, Normal Recall=0.8065, Normal Precision=0.9890, Attack Recall=0.9640, Attack Precision=0.5547

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
0.15       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819   <--
0.20       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.25       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.30       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.35       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.40       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.45       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.50       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.55       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.60       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.65       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.70       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.75       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.80       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8543, F1=0.7987, Normal Recall=0.8073, Normal Precision=0.9812, Attack Recall=0.9640, Attack Precision=0.6819

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
0.15       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689   <--
0.20       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.25       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.30       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.35       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.40       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.45       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.50       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.55       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.60       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.65       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.70       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.75       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.80       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8697, F1=0.8554, Normal Recall=0.8068, Normal Precision=0.9711, Attack Recall=0.9640, Attack Precision=0.7689

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
0.15       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318   <--
0.20       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.25       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.30       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.35       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.40       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.45       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.50       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.55       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.60       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.65       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.70       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.75       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.80       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8845, F1=0.8930, Normal Recall=0.8050, Normal Precision=0.9572, Attack Recall=0.9640, Attack Precision=0.8318

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
0.15       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562   <--
0.20       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.25       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.30       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.35       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.40       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.45       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.50       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.55       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.60       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.65       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.70       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.75       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
0.80       0.8221   0.5203   0.8063   0.9951   0.9645   0.3562  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8221, F1=0.5203, Normal Recall=0.8063, Normal Precision=0.9951, Attack Recall=0.9645, Attack Precision=0.3562

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
0.15       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547   <--
0.20       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.25       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.30       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.35       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.40       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.45       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.50       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.55       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.60       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.65       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.70       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.75       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
0.80       0.8380   0.7042   0.8065   0.9890   0.9640   0.5547  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.7042, Normal Recall=0.8065, Normal Precision=0.9890, Attack Recall=0.9640, Attack Precision=0.5547

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
0.15       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819   <--
0.20       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.25       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.30       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.35       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.40       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.45       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.50       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.55       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.60       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.65       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.70       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.75       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
0.80       0.8543   0.7987   0.8073   0.9812   0.9640   0.6819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8543, F1=0.7987, Normal Recall=0.8073, Normal Precision=0.9812, Attack Recall=0.9640, Attack Precision=0.6819

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
0.15       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689   <--
0.20       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.25       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.30       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.35       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.40       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.45       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.50       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.55       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.60       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.65       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.70       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.75       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
0.80       0.8697   0.8554   0.8068   0.9711   0.9640   0.7689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8697, F1=0.8554, Normal Recall=0.8068, Normal Precision=0.9711, Attack Recall=0.9640, Attack Precision=0.7689

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
0.15       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318   <--
0.20       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.25       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.30       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.35       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.40       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.45       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.50       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.55       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.60       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.65       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.70       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.75       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
0.80       0.8845   0.8930   0.8050   0.9572   0.9640   0.8318  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8845, F1=0.8930, Normal Recall=0.8050, Normal Precision=0.9572, Attack Recall=0.9640, Attack Precision=0.8318

```

