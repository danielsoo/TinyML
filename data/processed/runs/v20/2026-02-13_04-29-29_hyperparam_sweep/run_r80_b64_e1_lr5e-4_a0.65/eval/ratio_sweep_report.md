# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-16 06:26:55 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8554 | 0.8046 | 0.7531 | 0.7004 | 0.6487 | 0.5974 | 0.5454 | 0.4934 | 0.4425 | 0.3890 | 0.3381 |
| QAT+Prune only | 0.8619 | 0.8759 | 0.8885 | 0.9018 | 0.9153 | 0.9270 | 0.9415 | 0.9550 | 0.9664 | 0.9796 | 0.9928 |
| QAT+PTQ | 0.8618 | 0.8759 | 0.8885 | 0.9017 | 0.9153 | 0.9270 | 0.9414 | 0.9550 | 0.9663 | 0.9796 | 0.9928 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8618 | 0.8759 | 0.8885 | 0.9017 | 0.9153 | 0.9270 | 0.9414 | 0.9550 | 0.9663 | 0.9796 | 0.9928 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2569 | 0.3539 | 0.4037 | 0.4350 | 0.4565 | 0.4716 | 0.4830 | 0.4924 | 0.4990 | 0.5053 |
| QAT+Prune only | 0.0000 | 0.6155 | 0.7808 | 0.8584 | 0.9037 | 0.9315 | 0.9532 | 0.9686 | 0.9793 | 0.9887 | 0.9964 |
| QAT+PTQ | 0.0000 | 0.6154 | 0.7807 | 0.8583 | 0.9036 | 0.9315 | 0.9531 | 0.9686 | 0.9793 | 0.9887 | 0.9964 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6154 | 0.7807 | 0.8583 | 0.9036 | 0.9315 | 0.9531 | 0.9686 | 0.9793 | 0.9887 | 0.9964 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8554 | 0.8565 | 0.8568 | 0.8557 | 0.8557 | 0.8568 | 0.8564 | 0.8558 | 0.8599 | 0.8477 | 0.0000 |
| QAT+Prune only | 0.8619 | 0.8629 | 0.8624 | 0.8627 | 0.8637 | 0.8612 | 0.8644 | 0.8669 | 0.8611 | 0.8610 | 0.0000 |
| QAT+PTQ | 0.8618 | 0.8629 | 0.8624 | 0.8626 | 0.8637 | 0.8612 | 0.8644 | 0.8667 | 0.8606 | 0.8610 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8618 | 0.8629 | 0.8624 | 0.8626 | 0.8637 | 0.8612 | 0.8644 | 0.8667 | 0.8606 | 0.8610 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8554 | 0.0000 | 0.0000 | 0.0000 | 0.8554 | 1.0000 |
| 90 | 10 | 299,940 | 0.8046 | 0.2072 | 0.3377 | 0.2569 | 0.8565 | 0.9209 |
| 80 | 20 | 291,350 | 0.7531 | 0.3712 | 0.3381 | 0.3539 | 0.8568 | 0.8381 |
| 70 | 30 | 194,230 | 0.7004 | 0.5010 | 0.3381 | 0.4037 | 0.8557 | 0.7510 |
| 60 | 40 | 145,675 | 0.6487 | 0.6097 | 0.3381 | 0.4350 | 0.8557 | 0.6598 |
| 50 | 50 | 116,540 | 0.5974 | 0.7025 | 0.3381 | 0.4565 | 0.8568 | 0.5642 |
| 40 | 60 | 97,115 | 0.5454 | 0.7794 | 0.3381 | 0.4716 | 0.8564 | 0.4631 |
| 30 | 70 | 83,240 | 0.4934 | 0.8455 | 0.3381 | 0.4830 | 0.8558 | 0.3566 |
| 20 | 80 | 72,835 | 0.4425 | 0.9061 | 0.3381 | 0.4924 | 0.8599 | 0.2452 |
| 10 | 90 | 64,740 | 0.3890 | 0.9523 | 0.3381 | 0.4990 | 0.8477 | 0.1246 |
| 0 | 100 | 58,270 | 0.3381 | 1.0000 | 0.3381 | 0.5053 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8619 | 0.0000 | 0.0000 | 0.0000 | 0.8619 | 1.0000 |
| 90 | 10 | 299,940 | 0.8759 | 0.4460 | 0.9929 | 0.6155 | 0.8629 | 0.9991 |
| 80 | 20 | 291,350 | 0.8885 | 0.6434 | 0.9928 | 0.7808 | 0.8624 | 0.9979 |
| 70 | 30 | 194,230 | 0.9018 | 0.7561 | 0.9928 | 0.8584 | 0.8627 | 0.9964 |
| 60 | 40 | 145,675 | 0.9153 | 0.8292 | 0.9928 | 0.9037 | 0.8637 | 0.9945 |
| 50 | 50 | 116,540 | 0.9270 | 0.8774 | 0.9928 | 0.9315 | 0.8612 | 0.9917 |
| 40 | 60 | 97,115 | 0.9415 | 0.9166 | 0.9928 | 0.9532 | 0.8644 | 0.9876 |
| 30 | 70 | 83,240 | 0.9550 | 0.9456 | 0.9928 | 0.9686 | 0.8669 | 0.9810 |
| 20 | 80 | 72,835 | 0.9664 | 0.9662 | 0.9928 | 0.9793 | 0.8611 | 0.9676 |
| 10 | 90 | 64,740 | 0.9796 | 0.9847 | 0.9928 | 0.9887 | 0.8610 | 0.9299 |
| 0 | 100 | 58,270 | 0.9928 | 1.0000 | 0.9928 | 0.9964 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8618 | 0.0000 | 0.0000 | 0.0000 | 0.8618 | 1.0000 |
| 90 | 10 | 299,940 | 0.8759 | 0.4459 | 0.9929 | 0.6154 | 0.8629 | 0.9991 |
| 80 | 20 | 291,350 | 0.8885 | 0.6433 | 0.9928 | 0.7807 | 0.8624 | 0.9979 |
| 70 | 30 | 194,230 | 0.9017 | 0.7560 | 0.9928 | 0.8583 | 0.8626 | 0.9964 |
| 60 | 40 | 145,675 | 0.9153 | 0.8292 | 0.9928 | 0.9036 | 0.8637 | 0.9945 |
| 50 | 50 | 116,540 | 0.9270 | 0.8773 | 0.9928 | 0.9315 | 0.8612 | 0.9917 |
| 40 | 60 | 97,115 | 0.9414 | 0.9165 | 0.9928 | 0.9531 | 0.8644 | 0.9876 |
| 30 | 70 | 83,240 | 0.9550 | 0.9456 | 0.9928 | 0.9686 | 0.8667 | 0.9809 |
| 20 | 80 | 72,835 | 0.9663 | 0.9661 | 0.9928 | 0.9793 | 0.8606 | 0.9675 |
| 10 | 90 | 64,740 | 0.9796 | 0.9847 | 0.9928 | 0.9887 | 0.8610 | 0.9298 |
| 0 | 100 | 58,270 | 0.9928 | 1.0000 | 0.9928 | 0.9964 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8618 | 0.0000 | 0.0000 | 0.0000 | 0.8618 | 1.0000 |
| 90 | 10 | 299,940 | 0.8759 | 0.4459 | 0.9929 | 0.6154 | 0.8629 | 0.9991 |
| 80 | 20 | 291,350 | 0.8885 | 0.6433 | 0.9928 | 0.7807 | 0.8624 | 0.9979 |
| 70 | 30 | 194,230 | 0.9017 | 0.7560 | 0.9928 | 0.8583 | 0.8626 | 0.9964 |
| 60 | 40 | 145,675 | 0.9153 | 0.8292 | 0.9928 | 0.9036 | 0.8637 | 0.9945 |
| 50 | 50 | 116,540 | 0.9270 | 0.8773 | 0.9928 | 0.9315 | 0.8612 | 0.9917 |
| 40 | 60 | 97,115 | 0.9414 | 0.9165 | 0.9928 | 0.9531 | 0.8644 | 0.9876 |
| 30 | 70 | 83,240 | 0.9550 | 0.9456 | 0.9928 | 0.9686 | 0.8667 | 0.9809 |
| 20 | 80 | 72,835 | 0.9663 | 0.9661 | 0.9928 | 0.9793 | 0.8606 | 0.9675 |
| 10 | 90 | 64,740 | 0.9796 | 0.9847 | 0.9928 | 0.9887 | 0.8610 | 0.9298 |
| 0 | 100 | 58,270 | 0.9928 | 1.0000 | 0.9928 | 0.9964 | 0.0000 | 0.0000 |


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
0.15       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072   <--
0.20       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.25       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.30       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.35       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.40       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.45       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.50       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.55       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.60       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.65       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.70       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.75       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
0.80       0.8046   0.2567   0.8565   0.9209   0.3375   0.2072  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8046, F1=0.2567, Normal Recall=0.8565, Normal Precision=0.9209, Attack Recall=0.3375, Attack Precision=0.2072

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
0.15       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707   <--
0.20       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.25       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.30       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.35       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.40       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.45       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.50       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.55       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.60       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.65       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.70       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.75       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
0.80       0.7528   0.3536   0.8565   0.8381   0.3381   0.3707  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7528, F1=0.3536, Normal Recall=0.8565, Normal Precision=0.8381, Attack Recall=0.3381, Attack Precision=0.3707

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
0.15       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021   <--
0.20       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.25       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.30       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.35       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.40       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.45       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.50       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.55       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.60       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.65       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.70       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.75       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
0.80       0.7008   0.4041   0.8563   0.7512   0.3381   0.5021  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7008, F1=0.4041, Normal Recall=0.8563, Normal Precision=0.7512, Attack Recall=0.3381, Attack Precision=0.5021

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
0.15       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097   <--
0.20       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.25       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.30       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.35       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.40       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.45       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.50       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.55       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.60       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.65       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.70       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.75       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
0.80       0.6487   0.4350   0.8557   0.6598   0.3381   0.6097  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6487, F1=0.4350, Normal Recall=0.8557, Normal Precision=0.6598, Attack Recall=0.3381, Attack Precision=0.6097

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
0.15       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999   <--
0.20       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.25       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.30       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.35       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.40       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.45       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.50       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.55       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.60       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.65       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.70       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.75       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
0.80       0.5966   0.4559   0.8550   0.5636   0.3381   0.6999  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5966, F1=0.4559, Normal Recall=0.8550, Normal Precision=0.5636, Attack Recall=0.3381, Attack Precision=0.6999

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
0.15       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461   <--
0.20       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.25       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.30       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.35       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.40       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.45       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.50       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.55       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.60       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.65       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.70       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.75       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
0.80       0.8760   0.6157   0.8629   0.9991   0.9934   0.4461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8760, F1=0.6157, Normal Recall=0.8629, Normal Precision=0.9991, Attack Recall=0.9934, Attack Precision=0.4461

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
0.15       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446   <--
0.20       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.25       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.30       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.35       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.40       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.45       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.50       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.55       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.60       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.65       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.70       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.75       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
0.80       0.8891   0.7817   0.8631   0.9979   0.9928   0.6446  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8891, F1=0.7817, Normal Recall=0.8631, Normal Precision=0.9979, Attack Recall=0.9928, Attack Precision=0.6446

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
0.15       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558   <--
0.20       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.25       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.30       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.35       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.40       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.45       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.50       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.55       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.60       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.65       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.70       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.75       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
0.80       0.9016   0.8582   0.8625   0.9964   0.9928   0.7558  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9016, F1=0.8582, Normal Recall=0.8625, Normal Precision=0.9964, Attack Recall=0.9928, Attack Precision=0.7558

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
0.15       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275   <--
0.20       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.25       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.30       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.35       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.40       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.45       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.50       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.55       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.60       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.65       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.70       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.75       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
0.80       0.9143   0.9026   0.8620   0.9945   0.9928   0.8275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9143, F1=0.9026, Normal Recall=0.8620, Normal Precision=0.9945, Attack Recall=0.9928, Attack Precision=0.8275

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
0.15       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773   <--
0.20       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.25       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.30       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.35       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.40       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.45       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.50       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.55       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.60       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.65       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.70       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.75       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
0.80       0.9270   0.9315   0.8611   0.9917   0.9928   0.8773  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9270, F1=0.9315, Normal Recall=0.8611, Normal Precision=0.9917, Attack Recall=0.9928, Attack Precision=0.8773

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
0.15       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460   <--
0.20       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.25       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.30       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.35       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.40       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.45       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.50       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.55       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.60       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.65       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.70       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.75       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.80       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8759, F1=0.6156, Normal Recall=0.8629, Normal Precision=0.9991, Attack Recall=0.9934, Attack Precision=0.4460

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
0.15       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445   <--
0.20       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.25       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.30       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.35       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.40       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.45       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.50       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.55       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.60       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.65       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.70       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.75       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.80       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8891, F1=0.7816, Normal Recall=0.8631, Normal Precision=0.9979, Attack Recall=0.9928, Attack Precision=0.6445

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
0.15       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557   <--
0.20       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.25       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.30       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.35       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.40       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.45       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.50       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.55       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.60       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.65       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.70       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.75       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.80       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9015, F1=0.8582, Normal Recall=0.8624, Normal Precision=0.9964, Attack Recall=0.9928, Attack Precision=0.7557

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
0.15       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273   <--
0.20       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.25       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.30       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.35       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.40       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.45       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.50       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.55       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.60       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.65       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.70       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.75       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.80       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9142, F1=0.9025, Normal Recall=0.8618, Normal Precision=0.9944, Attack Recall=0.9928, Attack Precision=0.8273

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
0.15       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771   <--
0.20       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.25       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.30       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.35       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.40       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.45       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.50       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.55       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.60       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.65       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.70       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.75       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.80       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9268, F1=0.9314, Normal Recall=0.8609, Normal Precision=0.9917, Attack Recall=0.9928, Attack Precision=0.8771

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
0.15       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460   <--
0.20       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.25       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.30       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.35       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.40       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.45       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.50       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.55       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.60       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.65       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.70       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.75       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
0.80       0.8759   0.6156   0.8629   0.9991   0.9934   0.4460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8759, F1=0.6156, Normal Recall=0.8629, Normal Precision=0.9991, Attack Recall=0.9934, Attack Precision=0.4460

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
0.15       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445   <--
0.20       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.25       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.30       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.35       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.40       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.45       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.50       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.55       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.60       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.65       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.70       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.75       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
0.80       0.8891   0.7816   0.8631   0.9979   0.9928   0.6445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8891, F1=0.7816, Normal Recall=0.8631, Normal Precision=0.9979, Attack Recall=0.9928, Attack Precision=0.6445

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
0.15       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557   <--
0.20       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.25       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.30       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.35       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.40       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.45       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.50       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.55       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.60       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.65       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.70       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.75       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
0.80       0.9015   0.8582   0.8624   0.9964   0.9928   0.7557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9015, F1=0.8582, Normal Recall=0.8624, Normal Precision=0.9964, Attack Recall=0.9928, Attack Precision=0.7557

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
0.15       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273   <--
0.20       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.25       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.30       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.35       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.40       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.45       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.50       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.55       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.60       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.65       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.70       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.75       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
0.80       0.9142   0.9025   0.8618   0.9944   0.9928   0.8273  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9142, F1=0.9025, Normal Recall=0.8618, Normal Precision=0.9944, Attack Recall=0.9928, Attack Precision=0.8273

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
0.15       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771   <--
0.20       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.25       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.30       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.35       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.40       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.45       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.50       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.55       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.60       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.65       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.70       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.75       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
0.80       0.9268   0.9314   0.8609   0.9917   0.9928   0.8771  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9268, F1=0.9314, Normal Recall=0.8609, Normal Precision=0.9917, Attack Recall=0.9928, Attack Precision=0.8771

```

