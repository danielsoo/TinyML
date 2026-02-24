# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-21 12:11:14 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1560 | 0.2388 | 0.3227 | 0.4068 | 0.4910 | 0.5745 | 0.6571 | 0.7421 | 0.8255 | 0.9097 | 0.9927 |
| QAT+Prune only | 0.7504 | 0.7584 | 0.7662 | 0.7739 | 0.7820 | 0.7888 | 0.7981 | 0.8047 | 0.8140 | 0.8204 | 0.8293 |
| QAT+PTQ | 0.7509 | 0.7587 | 0.7664 | 0.7740 | 0.7820 | 0.7886 | 0.7978 | 0.8043 | 0.8135 | 0.8197 | 0.8285 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7509 | 0.7587 | 0.7664 | 0.7740 | 0.7820 | 0.7886 | 0.7978 | 0.8043 | 0.8135 | 0.8197 | 0.8285 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2070 | 0.3696 | 0.5010 | 0.6094 | 0.7000 | 0.7765 | 0.8435 | 0.9010 | 0.9519 | 0.9963 |
| QAT+Prune only | 0.0000 | 0.4063 | 0.5865 | 0.6876 | 0.7527 | 0.7970 | 0.8313 | 0.8560 | 0.8771 | 0.8926 | 0.9067 |
| QAT+PTQ | 0.0000 | 0.4065 | 0.5865 | 0.6874 | 0.7525 | 0.7967 | 0.8310 | 0.8556 | 0.8766 | 0.8921 | 0.9062 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4065 | 0.5865 | 0.6874 | 0.7525 | 0.7967 | 0.8310 | 0.8556 | 0.8766 | 0.8921 | 0.9062 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1560 | 0.1549 | 0.1551 | 0.1557 | 0.1566 | 0.1562 | 0.1536 | 0.1573 | 0.1566 | 0.1623 | 0.0000 |
| QAT+Prune only | 0.7504 | 0.7508 | 0.7504 | 0.7502 | 0.7505 | 0.7483 | 0.7514 | 0.7475 | 0.7530 | 0.7400 | 0.0000 |
| QAT+PTQ | 0.7509 | 0.7512 | 0.7508 | 0.7506 | 0.7509 | 0.7487 | 0.7519 | 0.7479 | 0.7533 | 0.7402 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7509 | 0.7512 | 0.7508 | 0.7506 | 0.7509 | 0.7487 | 0.7519 | 0.7479 | 0.7533 | 0.7402 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1560 | 0.0000 | 0.0000 | 0.0000 | 0.1560 | 1.0000 |
| 90 | 10 | 299,940 | 0.2388 | 0.1155 | 0.9936 | 0.2070 | 0.1549 | 0.9954 |
| 80 | 20 | 291,350 | 0.3227 | 0.2271 | 0.9927 | 0.3696 | 0.1551 | 0.9884 |
| 70 | 30 | 194,230 | 0.4068 | 0.3351 | 0.9927 | 0.5010 | 0.1557 | 0.9804 |
| 60 | 40 | 145,675 | 0.4910 | 0.4397 | 0.9927 | 0.6094 | 0.1566 | 0.9699 |
| 50 | 50 | 116,540 | 0.5745 | 0.5405 | 0.9927 | 0.7000 | 0.1562 | 0.9555 |
| 40 | 60 | 97,115 | 0.6571 | 0.6376 | 0.9927 | 0.7765 | 0.1536 | 0.9337 |
| 30 | 70 | 83,240 | 0.7421 | 0.7332 | 0.9927 | 0.8435 | 0.1573 | 0.9026 |
| 20 | 80 | 72,835 | 0.8255 | 0.8248 | 0.9927 | 0.9010 | 0.1566 | 0.8433 |
| 10 | 90 | 64,740 | 0.9097 | 0.9143 | 0.9927 | 0.9519 | 0.1623 | 0.7125 |
| 0 | 100 | 58,270 | 0.9927 | 1.0000 | 0.9927 | 0.9963 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7504 | 0.0000 | 0.0000 | 0.0000 | 0.7504 | 1.0000 |
| 90 | 10 | 299,940 | 0.7584 | 0.2693 | 0.8267 | 0.4063 | 0.7508 | 0.9750 |
| 80 | 20 | 291,350 | 0.7662 | 0.4537 | 0.8293 | 0.5865 | 0.7504 | 0.9462 |
| 70 | 30 | 194,230 | 0.7739 | 0.5873 | 0.8293 | 0.6876 | 0.7502 | 0.9111 |
| 60 | 40 | 145,675 | 0.7820 | 0.6890 | 0.8293 | 0.7527 | 0.7505 | 0.8683 |
| 50 | 50 | 116,540 | 0.7888 | 0.7671 | 0.8293 | 0.7970 | 0.7483 | 0.8142 |
| 40 | 60 | 97,115 | 0.7981 | 0.8334 | 0.8293 | 0.8313 | 0.7514 | 0.7458 |
| 30 | 70 | 83,240 | 0.8047 | 0.8846 | 0.8293 | 0.8560 | 0.7475 | 0.6523 |
| 20 | 80 | 72,835 | 0.8140 | 0.9307 | 0.8293 | 0.8771 | 0.7530 | 0.5244 |
| 10 | 90 | 64,740 | 0.8204 | 0.9663 | 0.8293 | 0.8926 | 0.7400 | 0.3251 |
| 0 | 100 | 58,270 | 0.8293 | 1.0000 | 0.8293 | 0.9067 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7509 | 0.0000 | 0.0000 | 0.0000 | 0.7509 | 1.0000 |
| 90 | 10 | 299,940 | 0.7587 | 0.2695 | 0.8262 | 0.4065 | 0.7512 | 0.9749 |
| 80 | 20 | 291,350 | 0.7664 | 0.4539 | 0.8285 | 0.5865 | 0.7508 | 0.9460 |
| 70 | 30 | 194,230 | 0.7740 | 0.5874 | 0.8285 | 0.6874 | 0.7506 | 0.9108 |
| 60 | 40 | 145,675 | 0.7820 | 0.6892 | 0.8285 | 0.7525 | 0.7509 | 0.8679 |
| 50 | 50 | 116,540 | 0.7886 | 0.7673 | 0.8285 | 0.7967 | 0.7487 | 0.8136 |
| 40 | 60 | 97,115 | 0.7978 | 0.8336 | 0.8285 | 0.8310 | 0.7519 | 0.7451 |
| 30 | 70 | 83,240 | 0.8043 | 0.8846 | 0.8285 | 0.8556 | 0.7479 | 0.6514 |
| 20 | 80 | 72,835 | 0.8135 | 0.9307 | 0.8285 | 0.8766 | 0.7533 | 0.5234 |
| 10 | 90 | 64,740 | 0.8197 | 0.9663 | 0.8285 | 0.8921 | 0.7402 | 0.3241 |
| 0 | 100 | 58,270 | 0.8285 | 1.0000 | 0.8285 | 0.9062 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7509 | 0.0000 | 0.0000 | 0.0000 | 0.7509 | 1.0000 |
| 90 | 10 | 299,940 | 0.7587 | 0.2695 | 0.8262 | 0.4065 | 0.7512 | 0.9749 |
| 80 | 20 | 291,350 | 0.7664 | 0.4539 | 0.8285 | 0.5865 | 0.7508 | 0.9460 |
| 70 | 30 | 194,230 | 0.7740 | 0.5874 | 0.8285 | 0.6874 | 0.7506 | 0.9108 |
| 60 | 40 | 145,675 | 0.7820 | 0.6892 | 0.8285 | 0.7525 | 0.7509 | 0.8679 |
| 50 | 50 | 116,540 | 0.7886 | 0.7673 | 0.8285 | 0.7967 | 0.7487 | 0.8136 |
| 40 | 60 | 97,115 | 0.7978 | 0.8336 | 0.8285 | 0.8310 | 0.7519 | 0.7451 |
| 30 | 70 | 83,240 | 0.8043 | 0.8846 | 0.8285 | 0.8556 | 0.7479 | 0.6514 |
| 20 | 80 | 72,835 | 0.8135 | 0.9307 | 0.8285 | 0.8766 | 0.7533 | 0.5234 |
| 10 | 90 | 64,740 | 0.8197 | 0.9663 | 0.8285 | 0.8921 | 0.7402 | 0.3241 |
| 0 | 100 | 58,270 | 0.8285 | 1.0000 | 0.8285 | 0.9062 | 0.0000 | 0.0000 |


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
0.15       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155   <--
0.20       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.25       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.30       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.35       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.40       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.45       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.50       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.55       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.60       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.65       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.70       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.75       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
0.80       0.2387   0.2068   0.1549   0.9948   0.9927   0.1155  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2387, F1=0.2068, Normal Recall=0.1549, Normal Precision=0.9948, Attack Recall=0.9927, Attack Precision=0.1155

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
0.15       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270   <--
0.20       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.25       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.30       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.35       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.40       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.45       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.50       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.55       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.60       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.65       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.70       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.75       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
0.80       0.3224   0.3695   0.1548   0.9884   0.9927   0.2270  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3224, F1=0.3695, Normal Recall=0.1548, Normal Precision=0.9884, Attack Recall=0.9927, Attack Precision=0.2270

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
0.15       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352   <--
0.20       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.25       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.30       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.35       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.40       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.45       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.50       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.55       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.60       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.65       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.70       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.75       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
0.80       0.4072   0.5012   0.1563   0.9804   0.9927   0.3352  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4072, F1=0.5012, Normal Recall=0.1563, Normal Precision=0.9804, Attack Recall=0.9927, Attack Precision=0.3352

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
0.15       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394   <--
0.20       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.25       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.30       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.35       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.40       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.45       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.50       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.55       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.60       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.65       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.70       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.75       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
0.80       0.4904   0.6092   0.1556   0.9698   0.9927   0.4394  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4904, F1=0.6092, Normal Recall=0.1556, Normal Precision=0.9698, Attack Recall=0.9927, Attack Precision=0.4394

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
0.15       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404   <--
0.20       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.25       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.30       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.35       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.40       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.45       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.50       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.55       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.60       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.65       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.70       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.75       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
0.80       0.5742   0.6998   0.1557   0.9553   0.9927   0.5404  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5742, F1=0.6998, Normal Recall=0.1557, Normal Precision=0.9553, Attack Recall=0.9927, Attack Precision=0.5404

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
0.15       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704   <--
0.20       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.25       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.30       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.35       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.40       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.45       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.50       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.55       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.60       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.65       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.70       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.75       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
0.80       0.7588   0.4081   0.7508   0.9757   0.8314   0.2704  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7588, F1=0.4081, Normal Recall=0.7508, Normal Precision=0.9757, Attack Recall=0.8314, Attack Precision=0.2704

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
0.15       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546   <--
0.20       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.25       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.30       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.35       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.40       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.45       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.50       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.55       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.60       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.65       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.70       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.75       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
0.80       0.7669   0.5873   0.7513   0.9462   0.8293   0.4546  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7669, F1=0.5873, Normal Recall=0.7513, Normal Precision=0.9462, Attack Recall=0.8293, Attack Precision=0.4546

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
0.15       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875   <--
0.20       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.25       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.30       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.35       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.40       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.45       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.50       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.55       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.60       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.65       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.70       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.75       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
0.80       0.7741   0.6877   0.7504   0.9112   0.8293   0.5875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7741, F1=0.6877, Normal Recall=0.7504, Normal Precision=0.9112, Attack Recall=0.8293, Attack Precision=0.5875

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
0.15       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885   <--
0.20       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.25       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.30       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.35       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.40       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.45       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.50       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.55       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.60       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.65       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.70       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.75       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
0.80       0.7817   0.7524   0.7499   0.8682   0.8293   0.6885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7817, F1=0.7524, Normal Recall=0.7499, Normal Precision=0.8682, Attack Recall=0.8293, Attack Precision=0.6885

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
0.15       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676   <--
0.20       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.25       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.30       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.35       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.40       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.45       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.50       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.55       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.60       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.65       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.70       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.75       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
0.80       0.7891   0.7972   0.7489   0.8144   0.8293   0.7676  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7891, F1=0.7972, Normal Recall=0.7489, Normal Precision=0.8144, Attack Recall=0.8293, Attack Precision=0.7676

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
0.15       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706   <--
0.20       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.25       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.30       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.35       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.40       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.45       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.50       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.55       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.60       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.65       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.70       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.75       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.80       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7592, F1=0.4083, Normal Recall=0.7512, Normal Precision=0.9756, Attack Recall=0.8308, Attack Precision=0.2706

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
0.15       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548   <--
0.20       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.25       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.30       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.35       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.40       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.45       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.50       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.55       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.60       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.65       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.70       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.75       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.80       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7671, F1=0.5872, Normal Recall=0.7517, Normal Precision=0.9460, Attack Recall=0.8285, Attack Precision=0.4548

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
0.15       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877   <--
0.20       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.25       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.30       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.35       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.40       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.45       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.50       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.55       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.60       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.65       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.70       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.75       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.80       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7742, F1=0.6876, Normal Recall=0.7509, Normal Precision=0.9108, Attack Recall=0.8285, Attack Precision=0.5877

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
0.15       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887   <--
0.20       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.25       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.30       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.35       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.40       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.45       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.50       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.55       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.60       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.65       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.70       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.75       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.80       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7522, Normal Recall=0.7503, Normal Precision=0.8678, Attack Recall=0.8285, Attack Precision=0.6887

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
0.15       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677   <--
0.20       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.25       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.30       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.35       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.40       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.45       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.50       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.55       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.60       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.65       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.70       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.75       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.80       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7889, F1=0.7969, Normal Recall=0.7493, Normal Precision=0.8137, Attack Recall=0.8285, Attack Precision=0.7677

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
0.15       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706   <--
0.20       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.25       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.30       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.35       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.40       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.45       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.50       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.55       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.60       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.65       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.70       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.75       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
0.80       0.7592   0.4083   0.7512   0.9756   0.8308   0.2706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7592, F1=0.4083, Normal Recall=0.7512, Normal Precision=0.9756, Attack Recall=0.8308, Attack Precision=0.2706

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
0.15       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548   <--
0.20       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.25       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.30       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.35       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.40       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.45       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.50       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.55       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.60       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.65       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.70       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.75       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
0.80       0.7671   0.5872   0.7517   0.9460   0.8285   0.4548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7671, F1=0.5872, Normal Recall=0.7517, Normal Precision=0.9460, Attack Recall=0.8285, Attack Precision=0.4548

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
0.15       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877   <--
0.20       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.25       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.30       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.35       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.40       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.45       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.50       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.55       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.60       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.65       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.70       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.75       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
0.80       0.7742   0.6876   0.7509   0.9108   0.8285   0.5877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7742, F1=0.6876, Normal Recall=0.7509, Normal Precision=0.9108, Attack Recall=0.8285, Attack Precision=0.5877

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
0.15       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887   <--
0.20       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.25       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.30       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.35       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.40       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.45       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.50       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.55       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.60       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.65       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.70       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.75       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
0.80       0.7816   0.7522   0.7503   0.8678   0.8285   0.6887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7522, Normal Recall=0.7503, Normal Precision=0.8678, Attack Recall=0.8285, Attack Precision=0.6887

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
0.15       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677   <--
0.20       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.25       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.30       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.35       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.40       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.45       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.50       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.55       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.60       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.65       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.70       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.75       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
0.80       0.7889   0.7969   0.7493   0.8137   0.8285   0.7677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7889, F1=0.7969, Normal Recall=0.7493, Normal Precision=0.8137, Attack Recall=0.8285, Attack Precision=0.7677

```

