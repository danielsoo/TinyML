# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-14 09:46:59 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8786 | 0.8906 | 0.9020 | 0.9143 | 0.9259 | 0.9362 | 0.9486 | 0.9608 | 0.9725 | 0.9834 | 0.9956 |
| QAT+Prune only | 0.7836 | 0.7993 | 0.8160 | 0.8331 | 0.8488 | 0.8649 | 0.8814 | 0.8983 | 0.9147 | 0.9310 | 0.9484 |
| QAT+PTQ | 0.7848 | 0.8004 | 0.8169 | 0.8339 | 0.8497 | 0.8653 | 0.8819 | 0.8985 | 0.9151 | 0.9312 | 0.9485 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7848 | 0.8004 | 0.8169 | 0.8339 | 0.8497 | 0.8653 | 0.8819 | 0.8985 | 0.9151 | 0.9312 | 0.9485 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6454 | 0.8025 | 0.8746 | 0.9149 | 0.9398 | 0.9587 | 0.9727 | 0.9830 | 0.9908 | 0.9978 |
| QAT+Prune only | 0.0000 | 0.4859 | 0.6733 | 0.7732 | 0.8338 | 0.8753 | 0.9056 | 0.9289 | 0.9468 | 0.9612 | 0.9735 |
| QAT+PTQ | 0.0000 | 0.4873 | 0.6745 | 0.7741 | 0.8347 | 0.8757 | 0.9060 | 0.9290 | 0.9470 | 0.9612 | 0.9736 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4873 | 0.6745 | 0.7741 | 0.8347 | 0.8757 | 0.9060 | 0.9290 | 0.9470 | 0.9612 | 0.9736 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8786 | 0.8789 | 0.8786 | 0.8795 | 0.8794 | 0.8769 | 0.8781 | 0.8797 | 0.8801 | 0.8740 | 0.0000 |
| QAT+Prune only | 0.7836 | 0.7827 | 0.7829 | 0.7837 | 0.7824 | 0.7813 | 0.7809 | 0.7815 | 0.7800 | 0.7746 | 0.0000 |
| QAT+PTQ | 0.7848 | 0.7839 | 0.7840 | 0.7848 | 0.7839 | 0.7822 | 0.7821 | 0.7820 | 0.7814 | 0.7753 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7848 | 0.7839 | 0.7840 | 0.7848 | 0.7839 | 0.7822 | 0.7821 | 0.7820 | 0.7814 | 0.7753 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8786 | 0.0000 | 0.0000 | 0.0000 | 0.8786 | 1.0000 |
| 90 | 10 | 299,940 | 0.8906 | 0.4774 | 0.9957 | 0.6454 | 0.8789 | 0.9995 |
| 80 | 20 | 291,350 | 0.9020 | 0.6721 | 0.9956 | 0.8025 | 0.8786 | 0.9987 |
| 70 | 30 | 194,230 | 0.9143 | 0.7798 | 0.9956 | 0.8746 | 0.8795 | 0.9979 |
| 60 | 40 | 145,675 | 0.9259 | 0.8462 | 0.9956 | 0.9149 | 0.8794 | 0.9967 |
| 50 | 50 | 116,540 | 0.9362 | 0.8899 | 0.9956 | 0.9398 | 0.8769 | 0.9950 |
| 40 | 60 | 97,115 | 0.9486 | 0.9245 | 0.9956 | 0.9587 | 0.8781 | 0.9925 |
| 30 | 70 | 83,240 | 0.9608 | 0.9508 | 0.9956 | 0.9727 | 0.8797 | 0.9884 |
| 20 | 80 | 72,835 | 0.9725 | 0.9708 | 0.9956 | 0.9830 | 0.8801 | 0.9803 |
| 10 | 90 | 64,740 | 0.9834 | 0.9861 | 0.9956 | 0.9908 | 0.8740 | 0.9566 |
| 0 | 100 | 58,270 | 0.9956 | 1.0000 | 0.9956 | 0.9978 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7836 | 0.0000 | 0.0000 | 0.0000 | 0.7836 | 1.0000 |
| 90 | 10 | 299,940 | 0.7993 | 0.3266 | 0.9485 | 0.4859 | 0.7827 | 0.9927 |
| 80 | 20 | 291,350 | 0.8160 | 0.5220 | 0.9484 | 0.6733 | 0.7829 | 0.9838 |
| 70 | 30 | 194,230 | 0.8331 | 0.6527 | 0.9484 | 0.7732 | 0.7837 | 0.9726 |
| 60 | 40 | 145,675 | 0.8488 | 0.7440 | 0.9484 | 0.8338 | 0.7824 | 0.9579 |
| 50 | 50 | 116,540 | 0.8649 | 0.8126 | 0.9484 | 0.8753 | 0.7813 | 0.9380 |
| 40 | 60 | 97,115 | 0.8814 | 0.8666 | 0.9484 | 0.9056 | 0.7809 | 0.9098 |
| 30 | 70 | 83,240 | 0.8983 | 0.9101 | 0.9484 | 0.9289 | 0.7815 | 0.8665 |
| 20 | 80 | 72,835 | 0.9147 | 0.9452 | 0.9484 | 0.9468 | 0.7800 | 0.7908 |
| 10 | 90 | 64,740 | 0.9310 | 0.9743 | 0.9484 | 0.9612 | 0.7746 | 0.6252 |
| 0 | 100 | 58,270 | 0.9484 | 1.0000 | 0.9484 | 0.9735 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7848 | 0.0000 | 0.0000 | 0.0000 | 0.7848 | 1.0000 |
| 90 | 10 | 299,940 | 0.8004 | 0.3278 | 0.9487 | 0.4873 | 0.7839 | 0.9928 |
| 80 | 20 | 291,350 | 0.8169 | 0.5233 | 0.9485 | 0.6745 | 0.7840 | 0.9838 |
| 70 | 30 | 194,230 | 0.8339 | 0.6538 | 0.9485 | 0.7741 | 0.7848 | 0.9726 |
| 60 | 40 | 145,675 | 0.8497 | 0.7453 | 0.9485 | 0.8347 | 0.7839 | 0.9580 |
| 50 | 50 | 116,540 | 0.8653 | 0.8132 | 0.9485 | 0.8757 | 0.7822 | 0.9382 |
| 40 | 60 | 97,115 | 0.8819 | 0.8672 | 0.9485 | 0.9060 | 0.7821 | 0.9101 |
| 30 | 70 | 83,240 | 0.8985 | 0.9103 | 0.9485 | 0.9290 | 0.7820 | 0.8668 |
| 20 | 80 | 72,835 | 0.9151 | 0.9455 | 0.9485 | 0.9470 | 0.7814 | 0.7914 |
| 10 | 90 | 64,740 | 0.9312 | 0.9743 | 0.9485 | 0.9612 | 0.7753 | 0.6257 |
| 0 | 100 | 58,270 | 0.9485 | 1.0000 | 0.9485 | 0.9736 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7848 | 0.0000 | 0.0000 | 0.0000 | 0.7848 | 1.0000 |
| 90 | 10 | 299,940 | 0.8004 | 0.3278 | 0.9487 | 0.4873 | 0.7839 | 0.9928 |
| 80 | 20 | 291,350 | 0.8169 | 0.5233 | 0.9485 | 0.6745 | 0.7840 | 0.9838 |
| 70 | 30 | 194,230 | 0.8339 | 0.6538 | 0.9485 | 0.7741 | 0.7848 | 0.9726 |
| 60 | 40 | 145,675 | 0.8497 | 0.7453 | 0.9485 | 0.8347 | 0.7839 | 0.9580 |
| 50 | 50 | 116,540 | 0.8653 | 0.8132 | 0.9485 | 0.8757 | 0.7822 | 0.9382 |
| 40 | 60 | 97,115 | 0.8819 | 0.8672 | 0.9485 | 0.9060 | 0.7821 | 0.9101 |
| 30 | 70 | 83,240 | 0.8985 | 0.9103 | 0.9485 | 0.9290 | 0.7820 | 0.8668 |
| 20 | 80 | 72,835 | 0.9151 | 0.9455 | 0.9485 | 0.9470 | 0.7814 | 0.7914 |
| 10 | 90 | 64,740 | 0.9312 | 0.9743 | 0.9485 | 0.9612 | 0.7753 | 0.6257 |
| 0 | 100 | 58,270 | 0.9485 | 1.0000 | 0.9485 | 0.9736 | 0.0000 | 0.0000 |


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
0.15       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775   <--
0.20       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.25       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.30       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.35       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.40       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.45       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.50       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.55       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.60       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.65       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.70       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.75       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
0.80       0.8906   0.6456   0.8789   0.9995   0.9961   0.4775  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8906, F1=0.6456, Normal Recall=0.8789, Normal Precision=0.9995, Attack Recall=0.9961, Attack Precision=0.4775

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
0.15       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732   <--
0.20       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.25       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.30       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.35       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.40       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.45       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.50       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.55       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.60       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.65       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.70       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.75       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
0.80       0.9024   0.8032   0.8792   0.9987   0.9956   0.6732  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9024, F1=0.8032, Normal Recall=0.8792, Normal Precision=0.9987, Attack Recall=0.9956, Attack Precision=0.6732

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
0.15       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793   <--
0.20       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.25       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.30       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.35       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.40       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.45       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.50       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.55       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.60       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.65       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.70       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.75       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
0.80       0.9141   0.8743   0.8792   0.9979   0.9956   0.7793  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9141, F1=0.8743, Normal Recall=0.8792, Normal Precision=0.9979, Attack Recall=0.9956, Attack Precision=0.7793

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
0.15       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459   <--
0.20       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.25       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.30       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.35       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.40       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.45       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.50       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.55       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.60       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.65       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.70       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.75       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
0.80       0.9257   0.9146   0.8790   0.9967   0.9956   0.8459  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9257, F1=0.9146, Normal Recall=0.8790, Normal Precision=0.9967, Attack Recall=0.9956, Attack Precision=0.8459

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
0.15       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916   <--
0.20       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.25       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.30       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.35       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.40       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.45       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.50       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.55       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.60       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.65       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.70       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.75       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
0.80       0.9373   0.9408   0.8790   0.9950   0.9956   0.8916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9373, F1=0.9408, Normal Recall=0.8790, Normal Precision=0.9950, Attack Recall=0.9956, Attack Precision=0.8916

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
0.15       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266   <--
0.20       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.25       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.30       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.35       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.40       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.45       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.50       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.55       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.60       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.65       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.70       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.75       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
0.80       0.7993   0.4859   0.7827   0.9928   0.9486   0.3266  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7993, F1=0.4859, Normal Recall=0.7827, Normal Precision=0.9928, Attack Recall=0.9486, Attack Precision=0.3266

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
0.15       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224   <--
0.20       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.25       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.30       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.35       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.40       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.45       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.50       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.55       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.60       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.65       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.70       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.75       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
0.80       0.8163   0.6737   0.7832   0.9838   0.9484   0.5224  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.6737, Normal Recall=0.7832, Normal Precision=0.9838, Attack Recall=0.9484, Attack Precision=0.5224

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
0.15       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529   <--
0.20       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.25       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.30       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.35       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.40       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.45       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.50       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.55       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.60       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.65       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.70       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.75       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
0.80       0.8332   0.7733   0.7839   0.9726   0.9484   0.6529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8332, F1=0.7733, Normal Recall=0.7839, Normal Precision=0.9726, Attack Recall=0.9484, Attack Precision=0.6529

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
0.15       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451   <--
0.20       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.25       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.30       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.35       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.40       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.45       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.50       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.55       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.60       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.65       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.70       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.75       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
0.80       0.8496   0.8345   0.7837   0.9579   0.9484   0.7451  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8496, F1=0.8345, Normal Recall=0.7837, Normal Precision=0.9579, Attack Recall=0.9484, Attack Precision=0.7451

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
0.15       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134   <--
0.20       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.25       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.30       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.35       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.40       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.45       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.50       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.55       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.60       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.65       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.70       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.75       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
0.80       0.8654   0.8757   0.7825   0.9381   0.9484   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8654, F1=0.8757, Normal Recall=0.7825, Normal Precision=0.9381, Attack Recall=0.9484, Attack Precision=0.8134

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
0.15       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278   <--
0.20       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.25       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.30       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.35       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.40       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.45       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.50       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.55       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.60       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.65       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.70       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.75       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.80       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8004, F1=0.4873, Normal Recall=0.7839, Normal Precision=0.9928, Attack Recall=0.9487, Attack Precision=0.3278

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
0.15       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239   <--
0.20       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.25       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.30       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.35       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.40       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.45       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.50       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.55       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.60       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.65       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.70       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.75       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.80       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.6749, Normal Recall=0.7845, Normal Precision=0.9838, Attack Recall=0.9485, Attack Precision=0.5239

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
0.15       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542   <--
0.20       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.25       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.30       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.35       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.40       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.45       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.50       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.55       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.60       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.65       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.70       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.75       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.80       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.7743, Normal Recall=0.7852, Normal Precision=0.9726, Attack Recall=0.9485, Attack Precision=0.6542

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
0.15       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461   <--
0.20       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.25       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.30       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.35       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.40       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.45       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.50       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.55       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.60       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.65       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.70       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.75       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.80       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8503, F1=0.8352, Normal Recall=0.7848, Normal Precision=0.9581, Attack Recall=0.9485, Attack Precision=0.7461

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
0.15       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143   <--
0.20       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.25       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.30       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.35       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.40       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.45       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.50       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.55       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.60       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.65       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.70       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.75       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.80       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8661, F1=0.8763, Normal Recall=0.7838, Normal Precision=0.9383, Attack Recall=0.9485, Attack Precision=0.8143

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
0.15       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278   <--
0.20       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.25       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.30       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.35       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.40       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.45       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.50       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.55       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.60       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.65       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.70       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.75       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
0.80       0.8004   0.4873   0.7839   0.9928   0.9487   0.3278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8004, F1=0.4873, Normal Recall=0.7839, Normal Precision=0.9928, Attack Recall=0.9487, Attack Precision=0.3278

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
0.15       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239   <--
0.20       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.25       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.30       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.35       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.40       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.45       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.50       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.55       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.60       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.65       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.70       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.75       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
0.80       0.8173   0.6749   0.7845   0.9838   0.9485   0.5239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.6749, Normal Recall=0.7845, Normal Precision=0.9838, Attack Recall=0.9485, Attack Precision=0.5239

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
0.15       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542   <--
0.20       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.25       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.30       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.35       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.40       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.45       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.50       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.55       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.60       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.65       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.70       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.75       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
0.80       0.8342   0.7743   0.7852   0.9726   0.9485   0.6542  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.7743, Normal Recall=0.7852, Normal Precision=0.9726, Attack Recall=0.9485, Attack Precision=0.6542

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
0.15       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461   <--
0.20       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.25       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.30       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.35       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.40       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.45       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.50       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.55       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.60       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.65       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.70       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.75       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
0.80       0.8503   0.8352   0.7848   0.9581   0.9485   0.7461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8503, F1=0.8352, Normal Recall=0.7848, Normal Precision=0.9581, Attack Recall=0.9485, Attack Precision=0.7461

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
0.15       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143   <--
0.20       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.25       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.30       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.35       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.40       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.45       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.50       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.55       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.60       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.65       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.70       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.75       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
0.80       0.8661   0.8763   0.7838   0.9383   0.9485   0.8143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8661, F1=0.8763, Normal Recall=0.7838, Normal Precision=0.9383, Attack Recall=0.9485, Attack Precision=0.8143

```

