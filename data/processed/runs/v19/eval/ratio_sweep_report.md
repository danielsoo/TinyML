# Ratio Sweep Report

| Item | Value |
|------|-------|
| **Model** | `models/tflite/saved_model_original.tflite` |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-10 21:17:44 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total ratios evaluated: 11

## Comparison Table

| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 100,000 | 0 | 0.7466 | 0.0000 | 0.0000 | 0.0000 | 0.7466 | 1.0000 |
| 90 | 10 | 299,940 | 269,946 | 29,994 | 0.7408 | 0.2336 | 0.6977 | 0.3500 | 0.7456 | 0.9569 |
| 80 | 20 | 291,350 | 233,080 | 58,270 | 0.7357 | 0.4063 | 0.6977 | 0.5136 | 0.7451 | 0.9079 |
| 70 | 30 | 194,230 | 135,961 | 58,269 | 0.7315 | 0.5407 | 0.6977 | 0.6092 | 0.7460 | 0.8520 |
| 60 | 40 | 145,675 | 87,405 | 58,270 | 0.7272 | 0.6476 | 0.6977 | 0.6717 | 0.7468 | 0.7875 |
| 50 | 50 | 116,540 | 58,270 | 58,270 | 0.7215 | 0.7326 | 0.6977 | 0.7147 | 0.7453 | 0.7115 |
| 40 | 60 | 97,115 | 38,846 | 58,269 | 0.7164 | 0.8038 | 0.6977 | 0.7470 | 0.7445 | 0.6215 |
| 30 | 70 | 83,240 | 24,972 | 58,268 | 0.7139 | 0.8676 | 0.6977 | 0.7735 | 0.7516 | 0.5159 |
| 20 | 80 | 72,835 | 14,567 | 58,268 | 0.7065 | 0.9153 | 0.6977 | 0.7919 | 0.7417 | 0.3802 |
| 10 | 90 | 64,740 | 6,474 | 58,266 | 0.7027 | 0.9612 | 0.6977 | 0.8086 | 0.7468 | 0.2154 |
| 0 | 100 | 58,270 | 0 | 58,270 | 0.6977 | 1.0000 | 0.6977 | 0.8220 | 0.0000 | 0.0000 |

## Detailed Metrics

### Normal 100% : Attack 0%

- **Test samples**: 100,000 (Normal=100,000, Attack=0)
- **Accuracy**: 0.7466
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **Normal Recall** (of actual normal, predicted normal): 0.7466
- **Normal Precision** (of predicted normal, actual normal): 1.0000

### Normal 90% : Attack 10%

- **Test samples**: 299,940 (Normal=269,946, Attack=29,994)
- **Accuracy**: 0.7408
- **Precision**: 0.2336
- **Recall**: 0.6977
- **F1-Score**: 0.3500
- **Normal Recall** (of actual normal, predicted normal): 0.7456
- **Normal Precision** (of predicted normal, actual normal): 0.9569

### Normal 80% : Attack 20%

- **Test samples**: 291,350 (Normal=233,080, Attack=58,270)
- **Accuracy**: 0.7357
- **Precision**: 0.4063
- **Recall**: 0.6977
- **F1-Score**: 0.5136
- **Normal Recall** (of actual normal, predicted normal): 0.7451
- **Normal Precision** (of predicted normal, actual normal): 0.9079

### Normal 70% : Attack 30%

- **Test samples**: 194,230 (Normal=135,961, Attack=58,269)
- **Accuracy**: 0.7315
- **Precision**: 0.5407
- **Recall**: 0.6977
- **F1-Score**: 0.6092
- **Normal Recall** (of actual normal, predicted normal): 0.7460
- **Normal Precision** (of predicted normal, actual normal): 0.8520

### Normal 60% : Attack 40%

- **Test samples**: 145,675 (Normal=87,405, Attack=58,270)
- **Accuracy**: 0.7272
- **Precision**: 0.6476
- **Recall**: 0.6977
- **F1-Score**: 0.6717
- **Normal Recall** (of actual normal, predicted normal): 0.7468
- **Normal Precision** (of predicted normal, actual normal): 0.7875

### Normal 50% : Attack 50%

- **Test samples**: 116,540 (Normal=58,270, Attack=58,270)
- **Accuracy**: 0.7215
- **Precision**: 0.7326
- **Recall**: 0.6977
- **F1-Score**: 0.7147
- **Normal Recall** (of actual normal, predicted normal): 0.7453
- **Normal Precision** (of predicted normal, actual normal): 0.7115

### Normal 40% : Attack 60%

- **Test samples**: 97,115 (Normal=38,846, Attack=58,269)
- **Accuracy**: 0.7164
- **Precision**: 0.8038
- **Recall**: 0.6977
- **F1-Score**: 0.7470
- **Normal Recall** (of actual normal, predicted normal): 0.7445
- **Normal Precision** (of predicted normal, actual normal): 0.6215

### Normal 30% : Attack 70%

- **Test samples**: 83,240 (Normal=24,972, Attack=58,268)
- **Accuracy**: 0.7139
- **Precision**: 0.8676
- **Recall**: 0.6977
- **F1-Score**: 0.7735
- **Normal Recall** (of actual normal, predicted normal): 0.7516
- **Normal Precision** (of predicted normal, actual normal): 0.5159

### Normal 20% : Attack 80%

- **Test samples**: 72,835 (Normal=14,567, Attack=58,268)
- **Accuracy**: 0.7065
- **Precision**: 0.9153
- **Recall**: 0.6977
- **F1-Score**: 0.7919
- **Normal Recall** (of actual normal, predicted normal): 0.7417
- **Normal Precision** (of predicted normal, actual normal): 0.3802

### Normal 10% : Attack 90%

- **Test samples**: 64,740 (Normal=6,474, Attack=58,266)
- **Accuracy**: 0.7027
- **Precision**: 0.9612
- **Recall**: 0.6977
- **F1-Score**: 0.8086
- **Normal Recall** (of actual normal, predicted normal): 0.7468
- **Normal Precision** (of predicted normal, actual normal): 0.2154

### Normal 0% : Attack 100%

- **Test samples**: 58,270 (Normal=0, Attack=58,270)
- **Accuracy**: 0.6977
- **Precision**: 1.0000
- **Recall**: 0.6977
- **F1-Score**: 0.8220
- **Normal Recall** (of actual normal, predicted normal): 0.0000
- **Normal Precision** (of predicted normal, actual normal): 0.0000


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
0.15       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337   <--
0.20       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.25       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.30       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.35       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.40       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.45       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.50       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.55       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.60       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.65       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.70       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.75       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
0.80       0.7409   0.3503   0.7456   0.9570   0.6985   0.2337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7409, F1=0.3503, Normal Recall=0.7456, Normal Precision=0.9570, Attack Recall=0.6985, Attack Precision=0.2337

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
0.15       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069   <--
0.20       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.25       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.30       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.35       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.40       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.45       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.50       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.55       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.60       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.65       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.70       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.75       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
0.80       0.7362   0.5140   0.7458   0.9080   0.6977   0.4069  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7362, F1=0.5140, Normal Recall=0.7458, Normal Precision=0.9080, Attack Recall=0.6977, Attack Precision=0.4069

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
0.15       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415   <--
0.20       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.25       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.30       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.35       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.40       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.45       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.50       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.55       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.60       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.65       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.70       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.75       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
0.80       0.7321   0.6098   0.7469   0.8522   0.6977   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7321, F1=0.6098, Normal Recall=0.7469, Normal Precision=0.8522, Attack Recall=0.6977, Attack Precision=0.5415

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
0.15       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483   <--
0.20       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.25       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.30       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.35       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.40       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.45       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.50       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.55       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.60       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.65       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.70       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.75       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
0.80       0.7277   0.6721   0.7476   0.7877   0.6977   0.6483  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7277, F1=0.6721, Normal Recall=0.7476, Normal Precision=0.7877, Attack Recall=0.6977, Attack Precision=0.6483

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
0.15       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347   <--
0.20       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.25       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.30       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.35       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.40       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.45       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.50       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.55       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.60       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.65       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.70       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.75       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
0.80       0.7229   0.7157   0.7481   0.7122   0.6977   0.7347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7229, F1=0.7157, Normal Recall=0.7481, Normal Precision=0.7122, Attack Recall=0.6977, Attack Precision=0.7347

```

