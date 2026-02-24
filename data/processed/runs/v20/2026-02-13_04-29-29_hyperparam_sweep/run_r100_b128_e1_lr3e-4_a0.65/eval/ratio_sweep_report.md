# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-20 23:12:38 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9194 | 0.9268 | 0.9333 | 0.9407 | 0.9471 | 0.9537 | 0.9611 | 0.9683 | 0.9745 | 0.9814 | 0.9882 |
| QAT+Prune only | 0.6458 | 0.6811 | 0.7158 | 0.7524 | 0.7876 | 0.8207 | 0.8577 | 0.8929 | 0.9282 | 0.9625 | 0.9983 |
| QAT+PTQ | 0.6460 | 0.6812 | 0.7159 | 0.7526 | 0.7877 | 0.8207 | 0.8578 | 0.8930 | 0.9283 | 0.9624 | 0.9983 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6460 | 0.6812 | 0.7159 | 0.7526 | 0.7877 | 0.8207 | 0.8578 | 0.8930 | 0.9283 | 0.9624 | 0.9983 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.7298 | 0.8556 | 0.9090 | 0.9373 | 0.9552 | 0.9683 | 0.9776 | 0.9842 | 0.9896 | 0.9941 |
| QAT+Prune only | 0.0000 | 0.3850 | 0.5842 | 0.7076 | 0.7899 | 0.8478 | 0.8938 | 0.9288 | 0.9570 | 0.9796 | 0.9992 |
| QAT+PTQ | 0.0000 | 0.3850 | 0.5843 | 0.7077 | 0.7900 | 0.8477 | 0.8939 | 0.9289 | 0.9570 | 0.9795 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3850 | 0.5843 | 0.7077 | 0.7900 | 0.8477 | 0.8939 | 0.9289 | 0.9570 | 0.9795 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9194 | 0.9199 | 0.9196 | 0.9203 | 0.9197 | 0.9191 | 0.9204 | 0.9218 | 0.9198 | 0.9194 | 0.0000 |
| QAT+Prune only | 0.6458 | 0.6459 | 0.6452 | 0.6470 | 0.6471 | 0.6431 | 0.6468 | 0.6470 | 0.6475 | 0.6401 | 0.0000 |
| QAT+PTQ | 0.6460 | 0.6459 | 0.6452 | 0.6472 | 0.6472 | 0.6431 | 0.6470 | 0.6473 | 0.6484 | 0.6392 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6460 | 0.6459 | 0.6452 | 0.6472 | 0.6472 | 0.6431 | 0.6470 | 0.6473 | 0.6484 | 0.6392 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9194 | 0.0000 | 0.0000 | 0.0000 | 0.9194 | 1.0000 |
| 90 | 10 | 299,940 | 0.9268 | 0.5784 | 0.9886 | 0.7298 | 0.9199 | 0.9986 |
| 80 | 20 | 291,350 | 0.9333 | 0.7544 | 0.9882 | 0.8556 | 0.9196 | 0.9968 |
| 70 | 30 | 194,230 | 0.9407 | 0.8416 | 0.9882 | 0.9090 | 0.9203 | 0.9946 |
| 60 | 40 | 145,675 | 0.9471 | 0.8913 | 0.9882 | 0.9373 | 0.9197 | 0.9916 |
| 50 | 50 | 116,540 | 0.9537 | 0.9243 | 0.9882 | 0.9552 | 0.9191 | 0.9874 |
| 40 | 60 | 97,115 | 0.9611 | 0.9491 | 0.9882 | 0.9683 | 0.9204 | 0.9812 |
| 30 | 70 | 83,240 | 0.9683 | 0.9672 | 0.9882 | 0.9776 | 0.9218 | 0.9711 |
| 20 | 80 | 72,835 | 0.9745 | 0.9801 | 0.9882 | 0.9842 | 0.9198 | 0.9514 |
| 10 | 90 | 64,740 | 0.9814 | 0.9910 | 0.9882 | 0.9896 | 0.9194 | 0.8968 |
| 0 | 100 | 58,270 | 0.9882 | 1.0000 | 0.9882 | 0.9941 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 0.6458 | 1.0000 |
| 90 | 10 | 299,940 | 0.6811 | 0.2385 | 0.9981 | 0.3850 | 0.6459 | 0.9997 |
| 80 | 20 | 291,350 | 0.7158 | 0.4129 | 0.9983 | 0.5842 | 0.6452 | 0.9993 |
| 70 | 30 | 194,230 | 0.7524 | 0.5480 | 0.9983 | 0.7076 | 0.6470 | 0.9989 |
| 60 | 40 | 145,675 | 0.7876 | 0.6535 | 0.9983 | 0.7899 | 0.6471 | 0.9983 |
| 50 | 50 | 116,540 | 0.8207 | 0.7367 | 0.9983 | 0.8478 | 0.6431 | 0.9974 |
| 40 | 60 | 97,115 | 0.8577 | 0.8092 | 0.9983 | 0.8938 | 0.6468 | 0.9961 |
| 30 | 70 | 83,240 | 0.8929 | 0.8684 | 0.9983 | 0.9288 | 0.6470 | 0.9940 |
| 20 | 80 | 72,835 | 0.9282 | 0.9189 | 0.9983 | 0.9570 | 0.6475 | 0.9897 |
| 10 | 90 | 64,740 | 0.9625 | 0.9615 | 0.9983 | 0.9796 | 0.6401 | 0.9769 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9992 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6460 | 0.0000 | 0.0000 | 0.0000 | 0.6460 | 1.0000 |
| 90 | 10 | 299,940 | 0.6812 | 0.2385 | 0.9981 | 0.3850 | 0.6459 | 0.9997 |
| 80 | 20 | 291,350 | 0.7159 | 0.4130 | 0.9983 | 0.5843 | 0.6452 | 0.9993 |
| 70 | 30 | 194,230 | 0.7526 | 0.5481 | 0.9983 | 0.7077 | 0.6472 | 0.9989 |
| 60 | 40 | 145,675 | 0.7877 | 0.6536 | 0.9983 | 0.7900 | 0.6472 | 0.9983 |
| 50 | 50 | 116,540 | 0.8207 | 0.7366 | 0.9983 | 0.8477 | 0.6431 | 0.9974 |
| 40 | 60 | 97,115 | 0.8578 | 0.8093 | 0.9983 | 0.8939 | 0.6470 | 0.9961 |
| 30 | 70 | 83,240 | 0.8930 | 0.8685 | 0.9983 | 0.9289 | 0.6473 | 0.9939 |
| 20 | 80 | 72,835 | 0.9283 | 0.9191 | 0.9983 | 0.9570 | 0.6484 | 0.9896 |
| 10 | 90 | 64,740 | 0.9624 | 0.9614 | 0.9983 | 0.9795 | 0.6392 | 0.9766 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6460 | 0.0000 | 0.0000 | 0.0000 | 0.6460 | 1.0000 |
| 90 | 10 | 299,940 | 0.6812 | 0.2385 | 0.9981 | 0.3850 | 0.6459 | 0.9997 |
| 80 | 20 | 291,350 | 0.7159 | 0.4130 | 0.9983 | 0.5843 | 0.6452 | 0.9993 |
| 70 | 30 | 194,230 | 0.7526 | 0.5481 | 0.9983 | 0.7077 | 0.6472 | 0.9989 |
| 60 | 40 | 145,675 | 0.7877 | 0.6536 | 0.9983 | 0.7900 | 0.6472 | 0.9983 |
| 50 | 50 | 116,540 | 0.8207 | 0.7366 | 0.9983 | 0.8477 | 0.6431 | 0.9974 |
| 40 | 60 | 97,115 | 0.8578 | 0.8093 | 0.9983 | 0.8939 | 0.6470 | 0.9961 |
| 30 | 70 | 83,240 | 0.8930 | 0.8685 | 0.9983 | 0.9289 | 0.6473 | 0.9939 |
| 20 | 80 | 72,835 | 0.9283 | 0.9191 | 0.9983 | 0.9570 | 0.6484 | 0.9896 |
| 10 | 90 | 64,740 | 0.9624 | 0.9614 | 0.9983 | 0.9795 | 0.6392 | 0.9766 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784   <--
0.20       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.25       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.30       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.35       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.40       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.45       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.50       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.55       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.60       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.65       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.70       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.75       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
0.80       0.9268   0.7299   0.9199   0.9987   0.9889   0.5784  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9268, F1=0.7299, Normal Recall=0.9199, Normal Precision=0.9987, Attack Recall=0.9889, Attack Precision=0.5784

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
0.15       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558   <--
0.20       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.25       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.30       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.35       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.40       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.45       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.50       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.55       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.60       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.65       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.70       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.75       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
0.80       0.9338   0.8565   0.9202   0.9968   0.9882   0.7558  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9338, F1=0.8565, Normal Recall=0.9202, Normal Precision=0.9968, Attack Recall=0.9882, Attack Precision=0.7558

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
0.15       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410   <--
0.20       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.25       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.30       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.35       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.40       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.45       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.50       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.55       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.60       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.65       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.70       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.75       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
0.80       0.9404   0.9087   0.9199   0.9946   0.9882   0.8410  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9404, F1=0.9087, Normal Recall=0.9199, Normal Precision=0.9946, Attack Recall=0.9882, Attack Precision=0.8410

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
0.15       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913   <--
0.20       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.25       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.30       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.35       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.40       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.45       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.50       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.55       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.60       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.65       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.70       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.75       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
0.80       0.9471   0.9373   0.9197   0.9916   0.9882   0.8913  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9471, F1=0.9373, Normal Recall=0.9197, Normal Precision=0.9916, Attack Recall=0.9882, Attack Precision=0.8913

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
0.15       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245   <--
0.20       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.25       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.30       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.35       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.40       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.45       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.50       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.55       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.60       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.65       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.70       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.75       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
0.80       0.9538   0.9553   0.9193   0.9874   0.9882   0.9245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9538, F1=0.9553, Normal Recall=0.9193, Normal Precision=0.9874, Attack Recall=0.9882, Attack Precision=0.9245

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
0.15       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386   <--
0.20       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.25       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.30       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.35       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.40       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.45       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.50       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.55       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.60       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.65       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.70       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.75       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
0.80       0.6811   0.3851   0.6459   0.9997   0.9985   0.2386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6811, F1=0.3851, Normal Recall=0.6459, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2386

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
0.15       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139   <--
0.20       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.25       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.30       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.35       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.40       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.45       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.50       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.55       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.60       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.65       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.70       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.75       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
0.80       0.7170   0.5852   0.6466   0.9994   0.9983   0.4139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7170, F1=0.5852, Normal Recall=0.6466, Normal Precision=0.9994, Attack Recall=0.9983, Attack Precision=0.4139

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
0.15       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474   <--
0.20       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.25       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.30       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.35       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.40       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.45       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.50       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.55       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.60       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.65       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.70       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.75       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
0.80       0.7518   0.7070   0.6462   0.9989   0.9983   0.5474  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.7070, Normal Recall=0.6462, Normal Precision=0.9989, Attack Recall=0.9983, Attack Precision=0.5474

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
0.15       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529   <--
0.20       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.25       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.30       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.35       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.40       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.45       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.50       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.55       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.60       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.65       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.70       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.75       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
0.80       0.7870   0.7895   0.6462   0.9983   0.9983   0.6529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7870, F1=0.7895, Normal Recall=0.6462, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.6529

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
0.15       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378   <--
0.20       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.25       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.30       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.35       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.40       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.45       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.50       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.55       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.60       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.65       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.70       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.75       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
0.80       0.8218   0.8485   0.6453   0.9974   0.9983   0.7378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8218, F1=0.8485, Normal Recall=0.6453, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.7378

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
0.15       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386   <--
0.20       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.25       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.30       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.35       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.40       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.45       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.50       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.55       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.60       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.65       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.70       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.75       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.80       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6812, F1=0.3851, Normal Recall=0.6460, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2386

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
0.15       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140   <--
0.20       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.25       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.30       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.35       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.40       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.45       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.50       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.55       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.60       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.65       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.70       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.75       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.80       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7170, F1=0.5853, Normal Recall=0.6467, Normal Precision=0.9993, Attack Recall=0.9983, Attack Precision=0.4140

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
0.15       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474   <--
0.20       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.25       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.30       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.35       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.40       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.45       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.50       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.55       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.60       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.65       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.70       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.75       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.80       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.7071, Normal Recall=0.6462, Normal Precision=0.9989, Attack Recall=0.9983, Attack Precision=0.5474

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
0.15       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530   <--
0.20       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.25       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.30       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.35       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.40       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.45       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.50       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.55       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.60       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.65       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.70       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.75       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.80       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7871, F1=0.7895, Normal Recall=0.6463, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.6530

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
0.15       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379   <--
0.20       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.25       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.30       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.35       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.40       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.45       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.50       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.55       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.60       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.65       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.70       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.75       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.80       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8218, F1=0.8485, Normal Recall=0.6453, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.7379

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
0.15       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386   <--
0.20       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.25       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.30       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.35       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.40       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.45       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.50       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.55       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.60       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.65       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.70       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.75       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
0.80       0.6812   0.3851   0.6460   0.9997   0.9985   0.2386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6812, F1=0.3851, Normal Recall=0.6460, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2386

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
0.15       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140   <--
0.20       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.25       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.30       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.35       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.40       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.45       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.50       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.55       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.60       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.65       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.70       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.75       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
0.80       0.7170   0.5853   0.6467   0.9993   0.9983   0.4140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7170, F1=0.5853, Normal Recall=0.6467, Normal Precision=0.9993, Attack Recall=0.9983, Attack Precision=0.4140

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
0.15       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474   <--
0.20       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.25       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.30       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.35       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.40       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.45       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.50       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.55       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.60       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.65       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.70       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.75       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
0.80       0.7518   0.7071   0.6462   0.9989   0.9983   0.5474  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.7071, Normal Recall=0.6462, Normal Precision=0.9989, Attack Recall=0.9983, Attack Precision=0.5474

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
0.15       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530   <--
0.20       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.25       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.30       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.35       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.40       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.45       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.50       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.55       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.60       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.65       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.70       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.75       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
0.80       0.7871   0.7895   0.6463   0.9983   0.9983   0.6530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7871, F1=0.7895, Normal Recall=0.6463, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.6530

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
0.15       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379   <--
0.20       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.25       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.30       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.35       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.40       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.45       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.50       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.55       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.60       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.65       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.70       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.75       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
0.80       0.8218   0.8485   0.6453   0.9974   0.9983   0.7379  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8218, F1=0.8485, Normal Recall=0.6453, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.7379

```

