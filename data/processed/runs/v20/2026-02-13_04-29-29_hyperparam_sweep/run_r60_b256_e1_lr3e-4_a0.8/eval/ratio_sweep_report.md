# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-15 06:51:12 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7128 | 0.7404 | 0.7682 | 0.7980 | 0.8263 | 0.8542 | 0.8836 | 0.9109 | 0.9401 | 0.9685 | 0.9975 |
| QAT+Prune only | 0.4622 | 0.5157 | 0.5685 | 0.6225 | 0.6755 | 0.7299 | 0.7836 | 0.8373 | 0.8912 | 0.9441 | 0.9984 |
| QAT+PTQ | 0.4627 | 0.5160 | 0.5688 | 0.6228 | 0.6758 | 0.7302 | 0.7836 | 0.8373 | 0.8914 | 0.9442 | 0.9984 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4627 | 0.5160 | 0.5688 | 0.6228 | 0.6758 | 0.7302 | 0.7836 | 0.8373 | 0.8914 | 0.9442 | 0.9984 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4346 | 0.6326 | 0.7476 | 0.8213 | 0.8724 | 0.9113 | 0.9400 | 0.9638 | 0.9828 | 0.9987 |
| QAT+Prune only | 0.0000 | 0.2919 | 0.4807 | 0.6134 | 0.7111 | 0.7871 | 0.8470 | 0.8958 | 0.9362 | 0.9698 | 0.9992 |
| QAT+PTQ | 0.0000 | 0.2921 | 0.4808 | 0.6136 | 0.7113 | 0.7872 | 0.8470 | 0.8957 | 0.9363 | 0.9699 | 0.9992 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2921 | 0.4808 | 0.6136 | 0.7113 | 0.7872 | 0.8470 | 0.8957 | 0.9363 | 0.9699 | 0.9992 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7128 | 0.7118 | 0.7109 | 0.7125 | 0.7122 | 0.7108 | 0.7127 | 0.7088 | 0.7105 | 0.7076 | 0.0000 |
| QAT+Prune only | 0.4622 | 0.4620 | 0.4611 | 0.4615 | 0.4602 | 0.4615 | 0.4614 | 0.4616 | 0.4626 | 0.4555 | 0.0000 |
| QAT+PTQ | 0.4627 | 0.4624 | 0.4614 | 0.4619 | 0.4607 | 0.4620 | 0.4615 | 0.4615 | 0.4635 | 0.4564 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4627 | 0.4624 | 0.4614 | 0.4619 | 0.4607 | 0.4620 | 0.4615 | 0.4615 | 0.4635 | 0.4564 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7128 | 0.0000 | 0.0000 | 0.0000 | 0.7128 | 1.0000 |
| 90 | 10 | 299,940 | 0.7404 | 0.2778 | 0.9977 | 0.4346 | 0.7118 | 0.9996 |
| 80 | 20 | 291,350 | 0.7682 | 0.4631 | 0.9975 | 0.6326 | 0.7109 | 0.9991 |
| 70 | 30 | 194,230 | 0.7980 | 0.5979 | 0.9975 | 0.7476 | 0.7125 | 0.9985 |
| 60 | 40 | 145,675 | 0.8263 | 0.6980 | 0.9975 | 0.8213 | 0.7122 | 0.9977 |
| 50 | 50 | 116,540 | 0.8542 | 0.7753 | 0.9975 | 0.8724 | 0.7108 | 0.9965 |
| 40 | 60 | 97,115 | 0.8836 | 0.8389 | 0.9975 | 0.9113 | 0.7127 | 0.9948 |
| 30 | 70 | 83,240 | 0.9109 | 0.8888 | 0.9975 | 0.9400 | 0.7088 | 0.9918 |
| 20 | 80 | 72,835 | 0.9401 | 0.9324 | 0.9975 | 0.9638 | 0.7105 | 0.9861 |
| 10 | 90 | 64,740 | 0.9685 | 0.9685 | 0.9975 | 0.9828 | 0.7076 | 0.9691 |
| 0 | 100 | 58,270 | 0.9975 | 1.0000 | 0.9975 | 0.9987 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4622 | 0.0000 | 0.0000 | 0.0000 | 0.4622 | 1.0000 |
| 90 | 10 | 299,940 | 0.5157 | 0.1710 | 0.9984 | 0.2919 | 0.4620 | 0.9996 |
| 80 | 20 | 291,350 | 0.5685 | 0.3165 | 0.9984 | 0.4807 | 0.4611 | 0.9991 |
| 70 | 30 | 194,230 | 0.6225 | 0.4427 | 0.9984 | 0.6134 | 0.4615 | 0.9985 |
| 60 | 40 | 145,675 | 0.6755 | 0.5522 | 0.9984 | 0.7111 | 0.4602 | 0.9976 |
| 50 | 50 | 116,540 | 0.7299 | 0.6496 | 0.9984 | 0.7871 | 0.4615 | 0.9965 |
| 40 | 60 | 97,115 | 0.7836 | 0.7355 | 0.9984 | 0.8470 | 0.4614 | 0.9947 |
| 30 | 70 | 83,240 | 0.8373 | 0.8123 | 0.9984 | 0.8958 | 0.4616 | 0.9918 |
| 20 | 80 | 72,835 | 0.8912 | 0.8814 | 0.9984 | 0.9362 | 0.4626 | 0.9861 |
| 10 | 90 | 64,740 | 0.9441 | 0.9429 | 0.9984 | 0.9698 | 0.4555 | 0.9688 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4627 | 0.0000 | 0.0000 | 0.0000 | 0.4627 | 1.0000 |
| 90 | 10 | 299,940 | 0.5160 | 0.1711 | 0.9984 | 0.2921 | 0.4624 | 0.9996 |
| 80 | 20 | 291,350 | 0.5688 | 0.3167 | 0.9984 | 0.4808 | 0.4614 | 0.9991 |
| 70 | 30 | 194,230 | 0.6228 | 0.4429 | 0.9984 | 0.6136 | 0.4619 | 0.9985 |
| 60 | 40 | 145,675 | 0.6758 | 0.5524 | 0.9984 | 0.7113 | 0.4607 | 0.9977 |
| 50 | 50 | 116,540 | 0.7302 | 0.6498 | 0.9984 | 0.7872 | 0.4620 | 0.9965 |
| 40 | 60 | 97,115 | 0.7836 | 0.7355 | 0.9984 | 0.8470 | 0.4615 | 0.9948 |
| 30 | 70 | 83,240 | 0.8373 | 0.8122 | 0.9984 | 0.8957 | 0.4615 | 0.9919 |
| 20 | 80 | 72,835 | 0.8914 | 0.8816 | 0.9984 | 0.9363 | 0.4635 | 0.9863 |
| 10 | 90 | 64,740 | 0.9442 | 0.9430 | 0.9984 | 0.9699 | 0.4564 | 0.9692 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4627 | 0.0000 | 0.0000 | 0.0000 | 0.4627 | 1.0000 |
| 90 | 10 | 299,940 | 0.5160 | 0.1711 | 0.9984 | 0.2921 | 0.4624 | 0.9996 |
| 80 | 20 | 291,350 | 0.5688 | 0.3167 | 0.9984 | 0.4808 | 0.4614 | 0.9991 |
| 70 | 30 | 194,230 | 0.6228 | 0.4429 | 0.9984 | 0.6136 | 0.4619 | 0.9985 |
| 60 | 40 | 145,675 | 0.6758 | 0.5524 | 0.9984 | 0.7113 | 0.4607 | 0.9977 |
| 50 | 50 | 116,540 | 0.7302 | 0.6498 | 0.9984 | 0.7872 | 0.4620 | 0.9965 |
| 40 | 60 | 97,115 | 0.7836 | 0.7355 | 0.9984 | 0.8470 | 0.4615 | 0.9948 |
| 30 | 70 | 83,240 | 0.8373 | 0.8122 | 0.9984 | 0.8957 | 0.4615 | 0.9919 |
| 20 | 80 | 72,835 | 0.8914 | 0.8816 | 0.9984 | 0.9363 | 0.4635 | 0.9863 |
| 10 | 90 | 64,740 | 0.9442 | 0.9430 | 0.9984 | 0.9699 | 0.4564 | 0.9692 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |


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
0.15       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778   <--
0.20       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.25       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.30       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.35       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.40       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.45       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.50       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.55       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.60       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.65       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.70       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.75       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
0.80       0.7404   0.4346   0.7118   0.9997   0.9978   0.2778  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7404, F1=0.4346, Normal Recall=0.7118, Normal Precision=0.9997, Attack Recall=0.9978, Attack Precision=0.2778

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
0.15       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645   <--
0.20       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.25       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.30       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.35       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.40       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.45       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.50       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.55       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.60       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.65       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.70       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.75       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
0.80       0.7695   0.6338   0.7125   0.9991   0.9975   0.4645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7695, F1=0.6338, Normal Recall=0.7125, Normal Precision=0.9991, Attack Recall=0.9975, Attack Precision=0.4645

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
0.15       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989   <--
0.20       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.25       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.30       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.35       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.40       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.45       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.50       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.55       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.60       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.65       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.70       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.75       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
0.80       0.7988   0.7484   0.7137   0.9985   0.9975   0.5989  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7988, F1=0.7484, Normal Recall=0.7137, Normal Precision=0.9985, Attack Recall=0.9975, Attack Precision=0.5989

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
0.15       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979   <--
0.20       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.25       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.30       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.35       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.40       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.45       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.50       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.55       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.60       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.65       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.70       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.75       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
0.80       0.8263   0.8213   0.7122   0.9977   0.9975   0.6979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8263, F1=0.8213, Normal Recall=0.7122, Normal Precision=0.9977, Attack Recall=0.9975, Attack Precision=0.6979

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
0.15       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765   <--
0.20       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.25       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.30       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.35       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.40       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.45       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.50       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.55       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.60       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.65       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.70       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.75       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
0.80       0.8552   0.8732   0.7129   0.9965   0.9975   0.7765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8552, F1=0.8732, Normal Recall=0.7129, Normal Precision=0.9965, Attack Recall=0.9975, Attack Precision=0.7765

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
0.15       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710   <--
0.20       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.25       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.30       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.35       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.40       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.45       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.50       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.55       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.60       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.65       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.70       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.75       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
0.80       0.5157   0.2919   0.4621   0.9996   0.9984   0.1710  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5157, F1=0.2919, Normal Recall=0.4621, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1710

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
0.15       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172   <--
0.20       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.25       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.30       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.35       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.40       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.45       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.50       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.55       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.60       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.65       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.70       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.75       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
0.80       0.5699   0.4815   0.4628   0.9991   0.9984   0.3172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5699, F1=0.4815, Normal Recall=0.4628, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.3172

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
0.15       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430   <--
0.20       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.25       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.30       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.35       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.40       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.45       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.50       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.55       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.60       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.65       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.70       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.75       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
0.80       0.6229   0.6137   0.4620   0.9985   0.9984   0.4430  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6229, F1=0.6137, Normal Recall=0.4620, Normal Precision=0.9985, Attack Recall=0.9984, Attack Precision=0.4430

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
0.15       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529   <--
0.20       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.25       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.30       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.35       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.40       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.45       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.50       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.55       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.60       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.65       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.70       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.75       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
0.80       0.6765   0.7117   0.4618   0.9977   0.9984   0.5529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6765, F1=0.7117, Normal Recall=0.4618, Normal Precision=0.9977, Attack Recall=0.9984, Attack Precision=0.5529

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
0.15       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496   <--
0.20       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.25       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.30       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.35       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.40       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.45       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.50       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.55       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.60       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.65       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.70       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.75       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
0.80       0.7299   0.7870   0.4614   0.9965   0.9984   0.6496  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7299, F1=0.7870, Normal Recall=0.4614, Normal Precision=0.9965, Attack Recall=0.9984, Attack Precision=0.6496

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
0.15       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711   <--
0.20       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.25       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.30       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.35       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.40       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.45       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.50       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.55       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.60       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.65       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.70       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.75       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.80       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5160, F1=0.2921, Normal Recall=0.4624, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1711

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
0.15       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174   <--
0.20       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.25       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.30       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.35       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.40       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.45       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.50       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.55       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.60       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.65       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.70       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.75       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.80       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5702, F1=0.4817, Normal Recall=0.4632, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.3174

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
0.15       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432   <--
0.20       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.25       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.30       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.35       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.40       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.45       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.50       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.55       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.60       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.65       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.70       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.75       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.80       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6232, F1=0.6139, Normal Recall=0.4624, Normal Precision=0.9985, Attack Recall=0.9984, Attack Precision=0.4432

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
0.15       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532   <--
0.20       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.25       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.30       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.35       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.40       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.45       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.50       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.55       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.60       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.65       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.70       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.75       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.80       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6767, F1=0.7119, Normal Recall=0.4623, Normal Precision=0.9977, Attack Recall=0.9984, Attack Precision=0.5532

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
0.15       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497   <--
0.20       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.25       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.30       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.35       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.40       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.45       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.50       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.55       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.60       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.65       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.70       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.75       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.80       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7301, F1=0.7872, Normal Recall=0.4618, Normal Precision=0.9965, Attack Recall=0.9984, Attack Precision=0.6497

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
0.15       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711   <--
0.20       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.25       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.30       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.35       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.40       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.45       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.50       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.55       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.60       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.65       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.70       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.75       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
0.80       0.5160   0.2921   0.4624   0.9996   0.9984   0.1711  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5160, F1=0.2921, Normal Recall=0.4624, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1711

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
0.15       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174   <--
0.20       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.25       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.30       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.35       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.40       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.45       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.50       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.55       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.60       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.65       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.70       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.75       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
0.80       0.5702   0.4817   0.4632   0.9991   0.9984   0.3174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5702, F1=0.4817, Normal Recall=0.4632, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.3174

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
0.15       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432   <--
0.20       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.25       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.30       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.35       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.40       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.45       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.50       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.55       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.60       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.65       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.70       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.75       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
0.80       0.6232   0.6139   0.4624   0.9985   0.9984   0.4432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6232, F1=0.6139, Normal Recall=0.4624, Normal Precision=0.9985, Attack Recall=0.9984, Attack Precision=0.4432

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
0.15       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532   <--
0.20       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.25       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.30       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.35       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.40       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.45       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.50       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.55       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.60       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.65       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.70       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.75       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
0.80       0.6767   0.7119   0.4623   0.9977   0.9984   0.5532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6767, F1=0.7119, Normal Recall=0.4623, Normal Precision=0.9977, Attack Recall=0.9984, Attack Precision=0.5532

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
0.15       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497   <--
0.20       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.25       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.30       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.35       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.40       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.45       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.50       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.55       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.60       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.65       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.70       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.75       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
0.80       0.7301   0.7872   0.4618   0.9965   0.9984   0.6497  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7301, F1=0.7872, Normal Recall=0.4618, Normal Precision=0.9965, Attack Recall=0.9984, Attack Precision=0.6497

```

