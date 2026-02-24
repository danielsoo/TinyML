# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-17 12:30:11 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1985 | 0.2775 | 0.3567 | 0.4362 | 0.5162 | 0.5949 | 0.6744 | 0.7544 | 0.8329 | 0.9132 | 0.9923 |
| QAT+Prune only | 0.6511 | 0.6860 | 0.7201 | 0.7545 | 0.7895 | 0.8234 | 0.8589 | 0.8938 | 0.9293 | 0.9623 | 0.9980 |
| QAT+PTQ | 0.6500 | 0.6848 | 0.7191 | 0.7537 | 0.7888 | 0.8227 | 0.8584 | 0.8934 | 0.9289 | 0.9622 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6500 | 0.6848 | 0.7191 | 0.7537 | 0.7888 | 0.8227 | 0.8584 | 0.8934 | 0.9289 | 0.9622 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2154 | 0.3816 | 0.5136 | 0.6214 | 0.7101 | 0.7853 | 0.8498 | 0.9048 | 0.9536 | 0.9962 |
| QAT+Prune only | 0.0000 | 0.3886 | 0.5878 | 0.7092 | 0.7914 | 0.8496 | 0.8946 | 0.9293 | 0.9576 | 0.9794 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.3877 | 0.5870 | 0.7085 | 0.7908 | 0.8491 | 0.8943 | 0.9291 | 0.9574 | 0.9794 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3877 | 0.5870 | 0.7085 | 0.7908 | 0.8491 | 0.8943 | 0.9291 | 0.9574 | 0.9794 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1985 | 0.1981 | 0.1978 | 0.1979 | 0.1988 | 0.1975 | 0.1975 | 0.1993 | 0.1951 | 0.2006 | 0.0000 |
| QAT+Prune only | 0.6511 | 0.6513 | 0.6506 | 0.6501 | 0.6505 | 0.6487 | 0.6501 | 0.6505 | 0.6544 | 0.6409 | 0.0000 |
| QAT+PTQ | 0.6500 | 0.6500 | 0.6493 | 0.6489 | 0.6492 | 0.6473 | 0.6490 | 0.6493 | 0.6524 | 0.6401 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6500 | 0.6500 | 0.6493 | 0.6489 | 0.6492 | 0.6473 | 0.6490 | 0.6493 | 0.6524 | 0.6401 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1985 | 0.0000 | 0.0000 | 0.0000 | 0.1985 | 1.0000 |
| 90 | 10 | 299,940 | 0.2775 | 0.1208 | 0.9920 | 0.2154 | 0.1981 | 0.9955 |
| 80 | 20 | 291,350 | 0.3567 | 0.2362 | 0.9923 | 0.3816 | 0.1978 | 0.9904 |
| 70 | 30 | 194,230 | 0.4362 | 0.3465 | 0.9923 | 0.5136 | 0.1979 | 0.9837 |
| 60 | 40 | 145,675 | 0.5162 | 0.4523 | 0.9923 | 0.6214 | 0.1988 | 0.9750 |
| 50 | 50 | 116,540 | 0.5949 | 0.5529 | 0.9923 | 0.7101 | 0.1975 | 0.9627 |
| 40 | 60 | 97,115 | 0.6744 | 0.6497 | 0.9923 | 0.7853 | 0.1975 | 0.9451 |
| 30 | 70 | 83,240 | 0.7544 | 0.7431 | 0.9923 | 0.8498 | 0.1993 | 0.9178 |
| 20 | 80 | 72,835 | 0.8329 | 0.8314 | 0.9923 | 0.9048 | 0.1951 | 0.8644 |
| 10 | 90 | 64,740 | 0.9132 | 0.9179 | 0.9923 | 0.9536 | 0.2006 | 0.7444 |
| 0 | 100 | 58,270 | 0.9923 | 1.0000 | 0.9923 | 0.9962 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6511 | 0.0000 | 0.0000 | 0.0000 | 0.6511 | 1.0000 |
| 90 | 10 | 299,940 | 0.6860 | 0.2412 | 0.9978 | 0.3886 | 0.6513 | 0.9996 |
| 80 | 20 | 291,350 | 0.7201 | 0.4166 | 0.9980 | 0.5878 | 0.6506 | 0.9992 |
| 70 | 30 | 194,230 | 0.7545 | 0.5501 | 0.9980 | 0.7092 | 0.6501 | 0.9987 |
| 60 | 40 | 145,675 | 0.7895 | 0.6556 | 0.9980 | 0.7914 | 0.6505 | 0.9980 |
| 50 | 50 | 116,540 | 0.8234 | 0.7396 | 0.9980 | 0.8496 | 0.6487 | 0.9969 |
| 40 | 60 | 97,115 | 0.8589 | 0.8106 | 0.9980 | 0.8946 | 0.6501 | 0.9954 |
| 30 | 70 | 83,240 | 0.8938 | 0.8695 | 0.9980 | 0.9293 | 0.6505 | 0.9929 |
| 20 | 80 | 72,835 | 0.9293 | 0.9203 | 0.9980 | 0.9576 | 0.6544 | 0.9880 |
| 10 | 90 | 64,740 | 0.9623 | 0.9616 | 0.9980 | 0.9794 | 0.6409 | 0.9728 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.6500 | 1.0000 |
| 90 | 10 | 299,940 | 0.6848 | 0.2406 | 0.9979 | 0.3877 | 0.6500 | 0.9996 |
| 80 | 20 | 291,350 | 0.7191 | 0.4157 | 0.9980 | 0.5870 | 0.6493 | 0.9992 |
| 70 | 30 | 194,230 | 0.7537 | 0.5492 | 0.9980 | 0.7085 | 0.6489 | 0.9987 |
| 60 | 40 | 145,675 | 0.7888 | 0.6548 | 0.9980 | 0.7908 | 0.6492 | 0.9980 |
| 50 | 50 | 116,540 | 0.8227 | 0.7389 | 0.9980 | 0.8491 | 0.6473 | 0.9970 |
| 40 | 60 | 97,115 | 0.8584 | 0.8101 | 0.9980 | 0.8943 | 0.6490 | 0.9955 |
| 30 | 70 | 83,240 | 0.8934 | 0.8691 | 0.9980 | 0.9291 | 0.6493 | 0.9930 |
| 20 | 80 | 72,835 | 0.9289 | 0.9199 | 0.9980 | 0.9574 | 0.6524 | 0.9881 |
| 10 | 90 | 64,740 | 0.9622 | 0.9615 | 0.9980 | 0.9794 | 0.6401 | 0.9732 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.6500 | 1.0000 |
| 90 | 10 | 299,940 | 0.6848 | 0.2406 | 0.9979 | 0.3877 | 0.6500 | 0.9996 |
| 80 | 20 | 291,350 | 0.7191 | 0.4157 | 0.9980 | 0.5870 | 0.6493 | 0.9992 |
| 70 | 30 | 194,230 | 0.7537 | 0.5492 | 0.9980 | 0.7085 | 0.6489 | 0.9987 |
| 60 | 40 | 145,675 | 0.7888 | 0.6548 | 0.9980 | 0.7908 | 0.6492 | 0.9980 |
| 50 | 50 | 116,540 | 0.8227 | 0.7389 | 0.9980 | 0.8491 | 0.6473 | 0.9970 |
| 40 | 60 | 97,115 | 0.8584 | 0.8101 | 0.9980 | 0.8943 | 0.6490 | 0.9955 |
| 30 | 70 | 83,240 | 0.8934 | 0.8691 | 0.9980 | 0.9291 | 0.6493 | 0.9930 |
| 20 | 80 | 72,835 | 0.9289 | 0.9199 | 0.9980 | 0.9574 | 0.6524 | 0.9881 |
| 10 | 90 | 64,740 | 0.9622 | 0.9615 | 0.9980 | 0.9794 | 0.6401 | 0.9732 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209   <--
0.20       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.25       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.30       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.35       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.40       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.45       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.50       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.55       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.60       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.65       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.70       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.75       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
0.80       0.2775   0.2155   0.1981   0.9957   0.9923   0.1209  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2775, F1=0.2155, Normal Recall=0.1981, Normal Precision=0.9957, Attack Recall=0.9923, Attack Precision=0.1209

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
0.15       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363   <--
0.20       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.25       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.30       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.35       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.40       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.45       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.50       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.55       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.60       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.65       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.70       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.75       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
0.80       0.3569   0.3817   0.1981   0.9904   0.9923   0.2363  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3569, F1=0.3817, Normal Recall=0.1981, Normal Precision=0.9904, Attack Recall=0.9923, Attack Precision=0.2363

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
0.15       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468   <--
0.20       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.25       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.30       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.35       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.40       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.45       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.50       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.55       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.60       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.65       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.70       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.75       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
0.80       0.4369   0.5139   0.1989   0.9838   0.9923   0.3468  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4369, F1=0.5139, Normal Recall=0.1989, Normal Precision=0.9838, Attack Recall=0.9923, Attack Precision=0.3468

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
0.15       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520   <--
0.20       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.25       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.30       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.35       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.40       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.45       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.50       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.55       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.60       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.65       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.70       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.75       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
0.80       0.5158   0.6211   0.1980   0.9749   0.9923   0.4520  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5158, F1=0.6211, Normal Recall=0.1980, Normal Precision=0.9749, Attack Recall=0.9923, Attack Precision=0.4520

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
0.15       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525   <--
0.20       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.25       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.30       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.35       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.40       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.45       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.50       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.55       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.60       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.65       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.70       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.75       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
0.80       0.5943   0.7098   0.1962   0.9625   0.9923   0.5525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5943, F1=0.7098, Normal Recall=0.1962, Normal Precision=0.9625, Attack Recall=0.9923, Attack Precision=0.5525

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
0.15       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413   <--
0.20       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.25       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.30       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.35       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.40       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.45       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.50       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.55       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.60       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.65       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.70       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.75       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
0.80       0.6860   0.3886   0.6513   0.9997   0.9981   0.2413  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6860, F1=0.3886, Normal Recall=0.6513, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2413

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
0.15       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175   <--
0.20       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.25       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.30       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.35       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.40       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.45       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.50       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.55       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.60       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.65       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.70       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.75       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
0.80       0.7211   0.5887   0.6519   0.9992   0.9980   0.4175  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7211, F1=0.5887, Normal Recall=0.6519, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4175

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
0.15       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508   <--
0.20       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.25       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.30       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.35       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.40       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.45       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.50       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.55       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.60       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.65       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.70       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.75       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
0.80       0.7553   0.7099   0.6512   0.9987   0.9980   0.5508  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7553, F1=0.7099, Normal Recall=0.6512, Normal Precision=0.9987, Attack Recall=0.9980, Attack Precision=0.5508

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
0.15       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559   <--
0.20       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.25       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.30       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.35       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.40       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.45       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.50       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.55       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.60       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.65       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.70       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.75       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
0.80       0.7897   0.7916   0.6509   0.9980   0.9980   0.6559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7897, F1=0.7916, Normal Recall=0.6509, Normal Precision=0.9980, Attack Recall=0.9980, Attack Precision=0.6559

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
0.15       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397   <--
0.20       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.25       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.30       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.35       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.40       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.45       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.50       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.55       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.60       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.65       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.70       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.75       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
0.80       0.8234   0.8497   0.6489   0.9969   0.9980   0.7397  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8234, F1=0.8497, Normal Recall=0.6489, Normal Precision=0.9969, Attack Recall=0.9980, Attack Precision=0.7397

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
0.15       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406   <--
0.20       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.25       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.30       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.35       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.40       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.45       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.50       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.55       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.60       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.65       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.70       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.75       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.80       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6848, F1=0.3878, Normal Recall=0.6500, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2406

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
0.15       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167   <--
0.20       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.25       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.30       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.35       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.40       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.45       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.50       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.55       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.60       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.65       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.70       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.75       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.80       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7202, F1=0.5879, Normal Recall=0.6507, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4167

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
0.15       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499   <--
0.20       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.25       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.30       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.35       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.40       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.45       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.50       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.55       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.60       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.65       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.70       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.75       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.80       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7543, F1=0.7091, Normal Recall=0.6499, Normal Precision=0.9987, Attack Recall=0.9980, Attack Precision=0.5499

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
0.15       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552   <--
0.20       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.25       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.30       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.35       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.40       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.45       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.50       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.55       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.60       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.65       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.70       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.75       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.80       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7891, F1=0.7911, Normal Recall=0.6499, Normal Precision=0.9980, Attack Recall=0.9980, Attack Precision=0.6552

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
0.15       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391   <--
0.20       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.25       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.30       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.35       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.40       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.45       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.50       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.55       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.60       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.65       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.70       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.75       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.80       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8229, F1=0.8493, Normal Recall=0.6477, Normal Precision=0.9970, Attack Recall=0.9980, Attack Precision=0.7391

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
0.15       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406   <--
0.20       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.25       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.30       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.35       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.40       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.45       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.50       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.55       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.60       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.65       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.70       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.75       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
0.80       0.6848   0.3878   0.6500   0.9997   0.9981   0.2406  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6848, F1=0.3878, Normal Recall=0.6500, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2406

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
0.15       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167   <--
0.20       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.25       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.30       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.35       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.40       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.45       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.50       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.55       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.60       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.65       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.70       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.75       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
0.80       0.7202   0.5879   0.6507   0.9992   0.9980   0.4167  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7202, F1=0.5879, Normal Recall=0.6507, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4167

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
0.15       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499   <--
0.20       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.25       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.30       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.35       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.40       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.45       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.50       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.55       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.60       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.65       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.70       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.75       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
0.80       0.7543   0.7091   0.6499   0.9987   0.9980   0.5499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7543, F1=0.7091, Normal Recall=0.6499, Normal Precision=0.9987, Attack Recall=0.9980, Attack Precision=0.5499

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
0.15       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552   <--
0.20       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.25       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.30       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.35       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.40       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.45       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.50       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.55       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.60       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.65       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.70       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.75       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
0.80       0.7891   0.7911   0.6499   0.9980   0.9980   0.6552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7891, F1=0.7911, Normal Recall=0.6499, Normal Precision=0.9980, Attack Recall=0.9980, Attack Precision=0.6552

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
0.15       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391   <--
0.20       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.25       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.30       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.35       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.40       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.45       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.50       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.55       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.60       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.65       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.70       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.75       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
0.80       0.8229   0.8493   0.6477   0.9970   0.9980   0.7391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8229, F1=0.8493, Normal Recall=0.6477, Normal Precision=0.9970, Attack Recall=0.9980, Attack Precision=0.7391

```

